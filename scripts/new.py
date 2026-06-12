import torch
import os
import cv2
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel
)
from diffusers.utils import load_image

# ==========================================================
# DEVICE
# ==========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ==========================================================
# PATHS
# ==========================================================
BASE_MODEL = "runwayml/stable-diffusion-v1-5"

LORA_PATH = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\lora_output\new setup\advtime_style_v1.safetensors"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "advtime_controlnet_result.png")

# Optional: input sketch / structure image
CONTROL_IMAGE_PATH = "control_input.png"  # put any image here

# ==========================================================
# LOAD CONTROLNET (CANNY FOR OUTLINES)
# ==========================================================
print("Loading ControlNet Canny...")

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_canny",
    torch_dtype=torch.float16
)

# ==========================================================
# LOAD PIPELINE
# ==========================================================
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None
)

pipe = pipe.to(device)

# Memory optimizations for RTX 3050 6GB
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()
pipe.enable_xformers_memory_efficient_attention()

print("Base model + ControlNet loaded ✅")

# ==========================================================
# LOAD LORA
# ==========================================================
print("Loading Adventure Time LoRA...")

pipe.load_lora_weights(LORA_PATH)
pipe.fuse_lora(lora_scale=0.85)

print("LoRA fused successfully ✅")

# ==========================================================
# PREPARE CONTROL IMAGE (CANNY EDGE)
# ==========================================================
print("Preparing Canny edges...")

init_image = load_image(CONTROL_IMAGE_PATH).convert("RGB")
init_image = init_image.resize((512, 512))

image_np = np.array(init_image)
edges = cv2.Canny(image_np, 100, 200)
edges = np.stack([edges]*3, axis=-1)
edges = Image.fromarray(edges)

# ==========================================================
# PROMPT
# ==========================================================
prompt = (
    "advtime_style, cartoon hero boy inside wooden treehouse, "
    "round head, small dot eyes, soft smile, "
    "flat pastel colors, thin clean black outlines, "
    "minimal shading, simple 2d western animation style, "
    "clean vector lineart, cel shading"
)

negative_prompt = (
    "realistic, photorealistic, anime, manga, 3d render, cgi, "
    "hyper detailed skin, dramatic shadows, complex textures"
)

# ==========================================================
# GENERATE
# ==========================================================
print("Generating image...")

with torch.autocast("cuda"):
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=edges,
        controlnet_conditioning_scale=1.0,
        num_inference_steps=25,
        guidance_scale=7,
        width=512,
        height=512
    ).images[0]

image.save(OUTPUT_PATH)

print("✅ Image saved at:", OUTPUT_PATH)