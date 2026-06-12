import torch
import os
from diffusers import StableDiffusionPipeline
from PIL import Image

# ----------------------------
# Device
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ----------------------------
# Load SD 1.5 (disable safety checker)
# ----------------------------
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None
)

pipe = pipe.to(device)

# ----------------------------
# Prompt (safe cartoon prompt)
# ----------------------------
prompt = (
    "classic cartoon style illustration, "
    "two cartoon characters running, "
    "bold black outlines, flat colors, "
    "comic panel"
)

# ----------------------------
# Generate image
# ----------------------------
image = pipe(
    prompt=prompt,
    num_inference_steps=25,
    guidance_scale=7.5
).images[0]

# ----------------------------
# Save output (AUTO CREATE FOLDER)
# ----------------------------
output_dir = "outputs/baseline"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "sd15_baseline.png")
image.save(output_path)

print("✅ Stable Diffusion baseline image saved at:", output_path)
