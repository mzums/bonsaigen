import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import time
from datetime import datetime
import math
import numpy as np
import re
import glob

from model import DataLoaderLite, GPTConfig, GPT, ConvEncoder, ConvDecoder


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

B = 32    # micro batch size
T = 4    # sequence length
total_batch_size = B * T * 4
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
model.class_weights = train_loader.class_weights.to(device)
#model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

# cosine learning rate decay
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 1000
#max_steps = 10000
max_steps = 10000
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


print("Class counts:", train_loader.class_counts)
print("Class weights:", train_loader.class_weights)


#import sys; sys.exit(0)


tree_dir = "../../dataset/tokenized/tree_0002"
T_start = 4
num_frames_to_generate = 200

frame_pattern = os.path.join(tree_dir, "frame_*.txt")
frame_files = glob.glob(frame_pattern)

def frame_id(path: str) -> int:
    match = re.search(r'frame_(\d+)\.txt$', path)
    return int(match.group(1)) if match else 0

frame_files = sorted(frame_files, key=frame_id)
print(len(frame_files))

initial_frames = []
for i in range(T_start):
    fname = frame_files[i]
    with open(fname, 'r') as f:
        content = f.read().strip()
    digits = re.sub(r'\s+', '', content)
    vec = np.array([int(c) for c in digits if c.isdigit()], dtype=np.float32)
    if len(vec) != 1152:
        if len(vec) < 1152:
            vec = np.pad(vec, (0, 1152 - len(vec)))
        else:
            vec = vec[:1152]
    initial_frames.append(vec)

context = np.stack(initial_frames, axis=0)  # (T_start, 1152)
context = torch.from_numpy(context).float().unsqueeze(0)  # (1, T_start, 1152)
context = context.to('cuda')

print(f"Context shape: {context.shape}")  # torch.Size([1, 4, 1152])

with torch.no_grad():
    # Use the first 4 frames as context, predict the 5th
    test_input = context   # (1, 4, 1152)
    logits, _ = model(test_input)
    pred = logits[:, -1, :, :].argmax(dim=-1)   # (1, 1152)

    # Load the true 5th frame (index 4) from the same dataset
    true_frame_path = frame_files[4]   # assuming your sorted list has it
    with open(true_frame_path, 'r') as f:
        content = f.read().strip()
    digits = re.sub(r'\s+', '', content)
    true_vec = np.array([int(c) for c in digits if c.isdigit()], dtype=np.float32)
    # apply the SAME normalization that you apply to context (if any)
    true_tensor = torch.from_numpy(true_vec).float().to(device)

    accuracy = (pred[0] == true_tensor).float().mean().item()
    print(f"Next‑frame prediction accuracy: {accuracy:.2%}")
    
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
generated = context

max_len = config.block_size
if context.shape[1] > max_len:
    raise ValueError(f"Context length {context.shape[1]} > block_size {max_len}")

with torch.no_grad():
    for step in range(num_frames_to_generate):
        if generated.shape[1] >= max_len:
            break

        logits, _ = model(generated)          # (1, T, 1152, 7)
        next_logits = logits[:, -1, :, :]     # (1, 1152, 7)

        # ---- top‑k first (no temperature) ----
        top_k = 5
        top_k_logits, top_k_indices = torch.topk(next_logits, top_k, dim=-1)

        masked_logits = torch.full_like(next_logits, float('-inf'))
        masked_logits.scatter_(-1, top_k_indices, top_k_logits)

        # ---- now apply temperature and clamp ----
        temperature = 0.6
        scaled_logits = masked_logits / temperature
        scaled_logits = torch.clamp(scaled_logits, min=-100, max=100)   # safe range

        # ---- softmax (no NaN) ----
        probs = torch.softmax(scaled_logits, dim=-1)   # (1, 1152, 7)

        # ---- sample ----
        probs_flat = probs.view(-1, 7)                 # (1152, 7)
        next_frame_indices = torch.multinomial(probs_flat, num_samples=1).view(1, 1152)
        next_frame = next_frame_indices.float().unsqueeze(1)
        generated = torch.cat([generated, next_frame], dim=1)
        

output_dir = "generated_frames"
os.makedirs(output_dir, exist_ok=True)

mapping = {
    0: ' ',
    1: '/',
    2: '|',
    3: '\\',
    4: '_',
    5: '~',
    6: '&',
}

num_frames = generated.shape[1]

for i in range(num_frames):
    frame_vec = generated[0, i, :].cpu().numpy()            # (1152,)                             # denormalize
    frame_vec = frame_vec.astype(int)
    
    grid = frame_vec.reshape(24, 48)  # (24, 48)
    
    def map_value(x):
        return mapping.get(x, str(x))
    
    """lines = []
    for row in range(24):
        line = ' '.join(map(str, grid[row]))
        lines.append(line)

    content = '\n'.join(lines)"""

    lines = []
    for row in range(24):
        line_chars = [map_value(grid[row, col]) for col in range(48)]
        lines.append(''.join(line_chars))

    content = '\n'.join(lines)
    
    filename = os.path.join(output_dir, f"frame_{i:04d}.txt")
    with open(filename, 'w') as f:
        f.write(content)
    
    if (i + 1) % 50 == 0:
        print(f"Saved {i+1}/{num_frames} frame")

print(f"✅ Saved all {num_frames} frames in directory: {output_dir}")

enc = ConvEncoder(n_emb=64)
dec = ConvDecoder(n_emb=64)
x = torch.randn(2, 1, 24, 48)
z = enc(x)          # (2, 64)
out = dec(z)        # (2, 7, 24, 48)
print(z.shape, out.shape)