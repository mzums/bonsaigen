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

class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.n_emb % config.n_head == 0
        self.c_attn = nn.Linear(config.n_emb, 3*config.n_emb)
        self.c_proj = nn.Linear(config.n_emb, config.n_emb)
        self.c_proj.NANOGPT_SCALE_INIT = 1
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
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_emb, 4*config.n_emb)
        # historical, it was like this in GPT2, so I use it
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_emb, config.n_emb)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        # norm before attention (unlike in the original transformer paper)
        self.ln_1 = nn.LayerNorm(config.n_emb)  # layer norm
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_emb)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))     # residual connection
        x = x + self.mlp(self.ln_2(x))      # residual connection
        return x

@dataclass
class GPTConfig:
    block_size: int = 256        # number of frames in context
    vocab_size: int = 1          # unused, but keeping for compatibility
    n_layer: int = 4
    n_head: int = 4
    n_emb: int = 64
    frame_dim: int = 24 * 48     # 1152


import torch.nn as nn

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # frame encoder: raw pixels (1152) -> embedding (n_emb)
        # linear for now
        self.frame_encoder = nn.Linear(config.frame_dim, config.n_emb, bias=False)

        # positional embeddings
        self.pos_embedding = nn.Embedding(config.block_size, config.n_emb)

        # transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # final layer norm and decoder
        self.ln_f = nn.LayerNorm(config.n_emb)
        self.frame_decoder = nn.Linear(config.n_emb, config.frame_dim, bias=False)

        # weight initialization
        self.apply(self._init_weights)

    def forward(self, x, targets=None):
        # x: (B, T, 1152)
        B, T, _ = x.shape
        
        # encode each frame to embedding space
        x = self.frame_encoder(x)  # (B, T, n_emb)
        
        # add positional embeddings
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)  # (1, T)
        x = x + self.pos_embedding(pos)  # (B, T, n_emb)
        
        # pass through transformer blocks
        for block in self.blocks:
            x = block(x)  # (B, T, n_emb)
        
        x = self.ln_f(x)
        logits = self.frame_decoder(x)  # (B, T, 1152)
        
        if targets is not None:
            loss = nn.functional.mse_loss(logits, targets)
            return logits, loss
        return logits, None
    
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


import tiktoken
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
                 data_root="../dataset/tokenized", val_frac=0.05):
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
                vec = vec / 9.0   # normalization
                all_frames.append(vec)

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
        y = torch.from_numpy(buf[1:]).float().view(B, T, -1)

        self.current_position += B * T * self.num_processes
        if self.current_position >= self.num_frames:
            self.current_position = (self.B * self.T * self.process_rank) % self.num_frames

        return x, y

# --------------------------------------

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


@torch.no_grad()
def evaluate_loss(model, val_loader, grad_accum_steps, device, ddp):
    model.eval()
    loss_accum = 0.0
    num_batches = 20  # num bathes toaverage
    val_loader.current_position = val_loader.B * val_loader.T * val_loader.process_rank
    for micro_step in range(num_batches):
        x, y = val_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / num_batches   # avg
        loss_accum += loss.detach()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    model.train()
    return loss_accum.item()


# DDP (distributed data parallel)
# torchrun sets variables RANK, LOCAL_RANK and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "CUDA needed for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0      # this process does logging, checkpointing etc
else:
    # non-ddp run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

B = 8    # micro batch size
T = 4    # sequence length
total_batch_size = B * T * 1
assert total_batch_size % (B * T * ddp_world_size) == 0, "total_batch_size shoudl be divisibel by B * T * ddp_world_size"
# gradient accumulation
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
val_loader   = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')

torch.set_float32_matmul_precision('medium')

# create model
#model = GPT.from_pretrained('gpt2')
#model = GPT(GPTConfig(vocab_size=50304))


config = GPTConfig()
model = GPT(config)
model.to(device)
#model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

# cosine learning rate decay
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 100
max_steps = 1000
def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr

    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)


for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps      # loss must be mean
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # set learning rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()    # wait for the gpu to finish work
    t1 = time.time()
    dt = (t1 - t0)
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process and step % 200 == 0:
        print(
            f"step {step}, loss: {loss_accum.item():.6f}, "
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            f"lr: {lr:.4e} norm: {norm:.4f} | "
            f"dt: {dt:.2f}s, tok/sec: {tokens_per_sec:.2f}"
        )

    if step % 1000 == 0 and master_process:
        val_loss = evaluate_loss(raw_model, val_loader, grad_accum_steps, device, ddp)
        print(f"\nstep {step} | validation loss: {val_loss:.6f}\n")


if master_process:
    torch.save({
    'model_state_dict': raw_model.state_dict(),
    'config': config
}, "bonsai_model.pt")

if ddp:
    destroy_process_group()


#import sys; sys.exit(0)
