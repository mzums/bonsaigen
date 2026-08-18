from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
import json
import time
import inspect
import os
from datetime import datetime

# ---------------------------------------------

class ConvEncoder(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding='same')
        self.conv2 = nn.Conv2d(32, 64, 3, padding='same')
        self.conv3 = nn.Conv2d(64, 128, 3, padding='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.maxpool = nn.MaxPool2d(2, 2)
        # Flatten: 128 * 6 * 12 = 9216
        self.fc1 = nn.Linear(128 * 6 * 12, 1024)
        self.fc2 = nn.Linear(1024, n_emb)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)               # 12x24
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)               # 6x12
        x = self.relu(self.bn3(self.conv3(x)))  # bez maxpool
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

class ConvDecoder(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.fc = nn.Linear(n_emb, 128 * 6 * 12)  # dostosuj kanały
        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)  # 6->12
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # 12->24
        self.conv1 = nn.Conv2d(64, 32, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(32, 7, kernel_size=3, padding='same')
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = x.reshape(-1, 128, 6, 12)
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.conv1(x))
        x = self.conv2(x)   # (B*T, 7, 24, 48)
        return x
    

class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.n_emb % config.n_head == 0
        self.c_attn = nn.Linear(config.n_emb, 3*config.n_emb)
        self.c_proj = nn.Linear(config.n_emb, config.n_emb)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.attn_dropout = nn.Dropout(config.dropout)
        self.proj_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_emb = config.n_emb
        # buffer because it should be constant, it's not a parameter
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
                            # (1,1) is because later we use self.bias[:,:,:T,:T]
                            # then we want to broadcast it to (batch_size, n_head)
        
    def forward(self, x):
        #print(f"x.shape: {x.shape}")
        # (batch_size, token_len, n_emb)
        B, T, C = x.size()
        # k,q,v are not learned, they are only acivations computed for every input
        # the model learns only weights in c_attn and weights in c_proj
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_emb, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # # for every query all the keys should sum to 1
        # att = F.softmax(att, dim=-1)
        # y = att @ v

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # to revert .transpose(1, 2) ^
        # transpose doesn't physically revert data, only changes metadata and shape but view requires contiguous data so we must physically revert the data
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.proj_dropout(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_emb, 4*config.n_emb)
        # historical, it was like this in GPT2, so I use it
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_emb, config.n_emb)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        # norm before attention (unlike in the original transformer paper)
        self.ln_1 = nn.LayerNorm(config.n_emb)  # layer norm
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_emb)
        self.mlp = MLP(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln_1(x)))     # residual connection
        x = x + self.dropout(self.mlp(self.ln_2(x)))      # residual connection
        return x

@dataclass
class GPTConfig:
    block_size: int = 256        # number of frames in context
    vocab_size: int = 7
    n_layer: int = 12
    n_head: int = 12
    n_emb: int = 768
    dropout: float = 0.2


import torch.nn as nn

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.frame_encoder = ConvEncoder(config.n_emb)
        self.pos_embedding = nn.Embedding(config.block_size, config.n_emb)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_emb)
        self.register_buffer('class_weights', torch.ones(config.vocab_size))
        
        self.frame_decoder = ConvDecoder(config.n_emb)
        self.apply(self._init_weights)

    def forward(self, x, targets=None):
        B, T, _ = x.shape
        x = x.view(B * T, 1, 24, 48)
        x = self.frame_encoder(x)          # (B*T, n_emb)
        x = x.view(B, T, -1)               # (B, T, n_emb)
        
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(pos)    # T musi być <= block_size
        
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)                   # (B, T, n_emb)
        
        x = x.view(B * T, -1)              # (B*T, n_emb)
        logits = self.frame_decoder(x)     # (B*T, 7, 24, 48)
        
        logits = logits.view(B, T, 7, 24, 48)
        logits = logits.permute(0, 1, 3, 4, 2)
        logits = logits.contiguous().view(B, T, -1, self.config.vocab_size)
        
        if targets is not None:
            targets = torch.clamp(targets, 0, self.config.vocab_size - 1)
            logits = logits / 2.0

            # Focal Loss
            main_loss = self.focal_loss(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                gamma=2.5,
                alpha=self.class_weights
            )

            # Progress loss
            preds = logits.argmax(dim=-1)
            density = (preds != 0).float().mean(dim=-1)
            decay_penalty = torch.relu(density[:, :-1] - density[:, 1:]).mean()
            jump_penalty = torch.relu((density[:, 1:] - density[:, :-1]) - 0.2).mean()
            progress_loss = decay_penalty + jump_penalty

            # Shape loss
            WOOD_CLASSES = [1, 2, 3, 4, 5]
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            entropy_penalty = -0.03 * entropy

            wood_prob = probs[..., WOOD_CLASSES].sum(dim=-1)
            wood_prob = wood_prob.view(B, T, 24, 48)

            grad_h = torch.abs(wood_prob[:, :, 1:, :] - wood_prob[:, :, :-1, :])
            grad_w = torch.abs(wood_prob[:, :, :, 1:] - wood_prob[:, :, :, :-1])
            mean_grad = (grad_h.mean() + grad_w.mean()) / 2.0
            shape_loss = torch.exp(-mean_grad * 8.0)
            #shape_loss = mean_grad

            # ---------

            up    = F.pad(wood_prob[:, :, :-1, :], (0, 0, 1, 0))
            down  = F.pad(wood_prob[:, :, 1:, :],  (0, 0, 0, 1))
            left  = F.pad(wood_prob[:, :, :, :-1], (1, 0, 0, 0))
            right = F.pad(wood_prob[:, :, :, 1:],  (0, 1, 0, 0))

            neighbor_max = torch.maximum(
                torch.maximum(up, down),
                torch.maximum(left, right)
            )


            isolated_loss = (
                wood_prob * F.relu(0.3 - neighbor_max)
            ).mean()



            # ---------

            wood_2d = wood_prob.reshape(B * T, 1, 24, 48)

            wood_count_5x5 = F.avg_pool2d(
                wood_2d,
                kernel_size=5,
                stride=1,
                padding=2
            ) * 25.0

            large_wood_loss = F.relu(wood_count_5x5 - 16.0).mean()


            loss = main_loss + 0.3 * shape_loss + 0.1 * progress_loss + 0.5 * entropy_penalty + 10.0 * isolated_loss + 0.05 * large_wood_loss

            return logits, loss
        return logits, None

    def focal_loss(self, logits, targets, gamma=2.0, alpha=None, label_smoothing=0.1):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=alpha, label_smoothing=label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.2)
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # overfitting happens because of multiplication, not addition
        # LayerNorm deletes bias after a linear layer anyway
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]       # weights
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]     # biases
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)} with {num_nodecay_params:,} parameters")

        # check if this version of AdamW takes fused as an argument
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        # fused combines a lot of small operations on ensors into a bigger operation
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


import torch

import numpy as np
import torch
import numpy as np
import torch
import os
import glob
import re
from typing import List, Optional
import numpy as np
import torch
import os
import glob
import re

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split='train',
                 data_root="../../dataset/tokenized", val_frac=0.05):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split

        tree_pattern = os.path.join(data_root, "tree_*")
        tree_dirs = glob.glob(tree_pattern)
        if not tree_dirs:
            raise ValueError(f"No tree directories found under {data_root}")

        def tree_id(path: str) -> int:
            match = re.search(r'tree_(\d+)$', path)
            return int(match.group(1)) if match else 0
        tree_dirs = sorted(tree_dirs, key=tree_id)

        split_idx = int(len(tree_dirs) * (1 - val_frac))
        if split == 'train':
            tree_dirs = tree_dirs[:split_idx]
        else:
            tree_dirs = tree_dirs[split_idx:]
        if not tree_dirs:
            raise ValueError(f"No tree directories for split '{split}'")

        all_frames = []
        for tree_dir in tree_dirs:
            frame_files = glob.glob(os.path.join(tree_dir, "frame_*.txt"))
            if not frame_files:
                continue
            def frame_id(path: str) -> int:
                match = re.search(r'frame_(\d+)\.txt$', path)
                return int(match.group(1)) if match else 0
            frame_files = sorted(frame_files, key=frame_id)

            for fname in frame_files:
                with open(fname, 'r') as f:
                    content = f.read().strip()
                digits = re.sub(r'\s+', '', content)
                vec = np.array([int(c) for c in digits if c.isdigit()], dtype=np.float32)
                if len(vec) != 24*48:
                    if len(vec) < 24*48:
                        vec = np.pad(vec, (0, 24*48 - len(vec)))
                    else:
                        vec = vec[:24*48]
                all_frames.append(vec.astype(np.float32))

        if not all_frames:
            raise ValueError(f"No frames loaded for split '{split}'")

        self.frames = np.stack(all_frames, axis=0)   # (N, 1152)
        self.num_frames = self.frames.shape[0]

        if self.num_frames < B * T + 1:
            raise ValueError(
                f"Dataset has only {self.num_frames} frames, "
                f"but batch needs {B*T+1} frames (B={B}, T={T}). Reduce B or T."
            )

        print(f"Loaded {self.num_frames} frames for {split} split")
        print(f"1 epoch = {self.num_frames // (B * T)} batches")

        self.class_counts = np.bincount(self.frames.flatten().astype(int), minlength=7)
        #self.class_weights = torch.tensor(1.0 / (self.class_counts + 1e-8), dtype=torch.float32)
        # smaller alpha is more spaces
        # alpha = 0.32 for 10k steps
        alpha = 0.18
        self.class_weights = torch.tensor(1.0 / np.power(self.class_counts + 1e-8, alpha), dtype=torch.float32)
        self.class_weights = self.class_weights / self.class_weights.mean()
        #self.class_weights = torch.ones(7, dtype=torch.float32)

        self.current_position = (self.B * self.T * self.process_rank) % self.num_frames

    def next_batch(self):
        B, T = self.B, self.T
        pos = self.current_position

        # wrapping
        if pos + B * T + 1 > self.num_frames:
            pos = (self.B * self.T * self.process_rank) % self.num_frames
            self.current_position = pos

        # Buffer of (B*T+1) frames
        buf = self.frames[pos : pos + B * T + 1]   # (B*T+1, 1152)
        assert buf.shape[0] == B * T + 1, f"Expected {B*T+1}, got {buf.shape[0]}"

        x = torch.from_numpy(buf[:-1]).float().view(B, T, -1)
        y = torch.from_numpy(buf[1:]).long().view(B, T, -1)

        self.current_position += B * T * self.num_processes
        if self.current_position >= self.num_frames:
            self.current_position = (self.B * self.T * self.process_rank) % self.num_frames

        return x, y
