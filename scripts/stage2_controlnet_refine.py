import torch
import os
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# ==========================================================
# DEVICE
# ==========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ==========================================================
# PATHS
# ==========================================================
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
CONTROLNET_MODEL = "lllyasviel/sd-controlnet-canny"

LORA_DIR = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\lora_output"
LORA_NAME = "PNT_ClassicStrip_Style_v1.safetensors"

STAGE1_IMAGE = os.path.join("outputs", "peanut_style", "peanut_stage45.png")

OUTPUT_DIR = os.path.join("outputs", "peanut_refined45")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "peanut_stage2.png")

# ==========================================================
# CHECK STAGE-1 IMAGE
# ==========================================================
if not os.path.exists(STAGE1_IMAGE):
    print("❌ Stage-1 image not found.")
    exit(1)

print("✅ Stage-1 image found:", STAGE1_IMAGE)

# ==========================================================
# LOAD CONTROLNET
# ==========================================================
controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_MODEL,
    torch_dtype=torch.float16
).to(device)

# ==========================================================
# LOAD PIPELINE
# ==========================================================
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

try:
    pipe.enable_xformers_memory_efficient_attention()
    print("xformers enabled.")
except:
    print("xformers not available.")

# ==========================================================
# LOAD LORA
# ==========================================================
pipe.load_lora_weights(LORA_DIR, weight_name=LORA_NAME)
pipe.fuse_lora(lora_scale=0.8)

print("✅ Peanut LoRA loaded successfully")

# ==========================================================
# LOAD STAGE-1 IMAGE
# ==========================================================
input_image = Image.open(STAGE1_IMAGE).convert("RGB")
input_image = input_image.resize((512, 512))

# ==========================================================
# CREATE CANNY EDGE MAP
# ==========================================================
np_img = np.array(input_image)
edges = cv2.Canny(np_img, 100, 200)

# Convert to 3-channel image (IMPORTANT)
edges = np.stack([edges] * 3, axis=-1)
edges = Image.fromarray(edges)

# ==========================================================
# PEANUT PROMPT
# ==========================================================
prompt = (
    "pnts_style, classic 1950s newspaper comic strip illustration, "
    "wide horizontal comic panel showing three sequential actions in one frame, "
    
    "left side: young girl cartoon character walking across green grass, "
    "short brown bob haircut, simple round head, small dot eyes, wearing green striped dress, "
    "neutral but determined expression, side profile walking pose, "
    
    "center: second young boy cartoon character facing her, round head, simple dot eyes, "
    "short spiky hair, wearing red shirt and dark shorts, pointing and laughing pose, "
    
    "right side: action moment where the girl punches the boy, "
    "boy recoiling backward with exaggerated cartoon motion, legs lifted slightly, "
    "large comic impact burst shape behind them, motion lines and stylized action stars, "
    
    "bold clean black outlines, flat pastel color palette, "
    "minimal shading, simple gradient sky background, "
    "clean green grass ground, balanced composition, "
    "2D hand-drawn vintage cartoon style"
)

negative_prompt = (
    "readable text, letters, words, typography, speech bubbles, dialogue, "
    "photorealistic, anime, manga, 3d render, cinematic lighting, "
    "realistic anatomy, detailed textures, glossy shading, "
    "watermark, logo, signature"
)

# ==========================================================
# GENERATE REFINED IMAGE
# ==========================================================
print("Refining Peanut image with ControlNet...")

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=edges,
    num_inference_steps=28,
    guidance_scale=6.8,
    controlnet_conditioning_scale=0.9,
    width=512,
    height=512
).images[0]

# ==========================================================
# SAVE
# ==========================================================
image.save(OUTPUT_PATH)

print("✅ Peanut Stage-2 image saved at:", OUTPUT_PATH)