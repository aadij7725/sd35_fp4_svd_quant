import os
import torch
from tqdm import tqdm

# Directory to save calibration latents
output_dir = "calib_data"
os.makedirs(output_dir, exist_ok=True)

# Number of calibration samples to generate
NUM_SAMPLES = 512
LATENT_SHAPE = (1, 4, 64, 64)  # Matches UNet input

print(f"Generating {NUM_SAMPLES} FP16 latent tensors for calibration...")

for i in tqdm(range(NUM_SAMPLES), desc="Saving latents"):
    latent = torch.randn(LATENT_SHAPE).half()
    torch.save(latent, os.path.join(output_dir, f"sample_{i}.pt"))

print(f"Saved {NUM_SAMPLES} calibration tensors in '{output_dir}/'")