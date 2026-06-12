import os
import torch
import cv2
import numpy as np
from PIL import Image

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler
)

# ==========================================================
# OFFLINE MODE
# ==========================================================

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ==========================================================
# DEVICE
# ==========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ==========================================================
# PATHS
# ==========================================================

BASE_MODEL = r"C:\AI\models\dreamshaper8"

CONTROLNET_MODEL = r"C:\AI\models\controlnet-canny"

LORA_PATH = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\lora_output\linart\abhi_pencil_portrait_lora.safetensors"

INPUT_IMAGE = r"input_photo.jpg"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "pencil_controlnet.png")

# ==========================================================
# LOAD CONTROLNET
# ==========================================================

print("Loading ControlNet...")

controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_MODEL,
    torch_dtype=torch.float16,
    local_files_only=True
)

# ==========================================================
# LOAD PIPELINE
# ==========================================================

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None,
    local_files_only=True
)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config
)

pipe = pipe.to(device)

pipe.enable_attention_slicing()
pipe.vae.enable_slicing()

print("Base model loaded")

# ==========================================================
# LOAD LORA
# ==========================================================

print("Loading LoRA...")

pipe.load_lora_weights(LORA_PATH)

pipe.fuse_lora(lora_scale=1.3)

print("LoRA loaded")

# ==========================================================
# LOAD PHOTO
# ==========================================================

image = cv2.imread(INPUT_IMAGE)

image = cv2.resize(image, (512, 512))

# ==========================================================
# CANNY EDGE DETECTION
# ==========================================================

edges = cv2.Canny(image, 100, 200)

edges = np.stack([edges]*3, axis=2)

control_image = Image.fromarray(edges)

# ==========================================================
# PROMPTS
# ==========================================================

prompt = (
    "masterpiece, best quality, "
    "(abhi_pencil_portrait:1.4), "
    "realistic graphite pencil sketch, "
    "portrait drawing, cross hatching shading, "
    "highly detailed pencil drawing"
)

negative_prompt = (
    "color, painting, anime, cartoon, "
    "3d render, cgi, "
    "paper border, notebook page, "
    "blurry, low quality, watermark"
)

# ==========================================================
# GENERATE IMAGE
# ==========================================================

generator = torch.Generator(device=device).manual_seed(123)

with torch.autocast(device):
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control_image,
        num_inference_steps=30,
        guidance_scale=7.5,
        width=512,
        height=512,
        generator=generator
    ).images[0]

# ==========================================================
# SAVE
# ==========================================================

image.save(OUTPUT_PATH)

print("Saved:", OUTPUT_PATH)