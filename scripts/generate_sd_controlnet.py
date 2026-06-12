import torch
import os
import cv2
import numpy as np
from PIL import Image

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel
)

# ----------------------------
# Device
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ----------------------------
# Load ControlNet (Canny)
# ----------------------------
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)

# ----------------------------
# Load SD 1.5 + ControlNet
# ----------------------------
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)

# ----------------------------
# Load input image (RELATIVE PATH ✔)
# ----------------------------
input_path = os.path.join("inputs", "sd15_baseline.png")

if not os.path.exists(input_path):
    raise FileNotFoundError(
        f"❌ Input image not found at {input_path}"
    )

input_image = Image.open(input_path).convert("RGB")

# ----------------------------
# Convert to Canny edges
# ----------------------------
np_img = np.array(input_image)
edges = cv2.Canny(np_img, 100, 200)
edges = Image.fromarray(edges)

# ----------------------------
# Prompt
# ----------------------------
prompt = (
    "classic cartoon style illustration, "
    "slapstick action, bold black outlines, "
    "flat vintage colors, comic panel"
)

# ----------------------------
# Generate image
# ----------------------------
image = pipe(
    prompt=prompt,
    image=edges,
    num_inference_steps=25,
    guidance_scale=7.5,
    controlnet_conditioning_scale=1.0
).images[0]

# ----------------------------
# Save output
# ----------------------------
output_dir = os.path.join("outputs", "controlnet")
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "sd15_controlnet.png")
image.save(output_path)

print("✅ ControlNet output saved at:", output_path)
