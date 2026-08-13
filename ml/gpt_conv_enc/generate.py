
import os
import glob
import re
import numpy as np
import torch

from model import GPT, ConvEncoder, ConvDecoder

def load_checkpoint(path):
    print(f"Loading model from: {path}")

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)

    print("\nCheckpoint contents:")
    print("  Keys:", list(checkpoint.keys()))

    return checkpoint


tree_dir = "../../dataset/tokenized/tree_0003"
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

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = load_checkpoint("bonsai_model.pt")
config = checkpoint['config']
print("\nCreating model...")
model = GPT(config)
model.load_state_dict(checkpoint['model_state_dict'])
for name, param in model.named_parameters():
    print(f"{name}: mean={param.mean():.6f}, std={param.std():.6f}")
    break

model.to(device)

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

        top_k = 2
        top_k_logits, top_k_indices = torch.topk(next_logits, top_k, dim=-1)

        masked_logits = torch.full_like(next_logits, float('-inf'))
        masked_logits.scatter_(-1, top_k_indices, top_k_logits)

        temperature = 1.5
        scaled_logits = masked_logits / temperature
        scaled_logits = torch.clamp(scaled_logits, min=-100, max=100)

        probs = torch.softmax(scaled_logits, dim=-1)   # (1, 1152, 7)

        probs_flat = probs.view(-1, 7)                 # (1152, 7)
        next_frame_indices = torch.multinomial(probs_flat, num_samples=1).view(1, 1152)
        next_frame = next_frame_indices.float().unsqueeze(1)
        generated = torch.cat([generated, next_frame], dim=1)
        

output_dir = "generated_frames3"
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