import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
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
# Load ControlNet
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
# Load LoRA
# ----------------------------
lora_path = os.path.join("lora", "cartoon_style_lora.safetensors")

if not os.path.exists(lora_path):
    raise FileNotFoundError(f"❌ LoRA not found at {lora_path}")

pipe.load_lora_weights(lora_path)
pipe.fuse_lora()
print("✅ LoRA loaded")

# ----------------------------
# Load input image
# ----------------------------
input_path = os.path.join("inputs", "sd15_baseline.png")

if not os.path.exists(input_path):
    raise FileNotFoundError(f"❌ Input image not found at {input_path}")

input_image = Image.open(input_path).convert("RGB")

# ----------------------------
# Canny edges
# ----------------------------
edges = cv2.Canny(np.array(input_image), 100, 200)
edges = Image.fromarray(edges)

# ----------------------------
# Prompt
# ----------------------------
prompt = (
    "classic cartoon style illustration, "
    "bold black outlines, flat colors, "
    "hand drawn animation, comic panel"
)

# ----------------------------
# Generate
# ----------------------------
image = pipe(
    prompt=prompt,
    image=edges,
    num_inference_steps=30,
    guidance_scale=8.0,
    controlnet_conditioning_scale=1.0
).images[0]

# ----------------------------
# Save
# ----------------------------
out_dir = os.path.join("outputs", "controlnet_lora")
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(out_dir, "sd15_controlnet_lora.png")
image.save(out_path)

print("✅ Final output saved at:", out_path)
