import torch
import os
from diffusers import StableDiffusionPipeline
from PIL import Image

# ==========================================================
# DEVICE
# ==========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ==========================================================
# PATH CONFIG
# ==========================================================
BASE_MODEL = "runwayml/stable-diffusion-v1-5"

LORA_DIR = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\lora_output"
LORA_NAME = "ClayAnimationRedmond15-ClayAnimation-Clay.safetensors"

OUTPUT_DIR = os.path.join("outputs", "peanut_style")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "peanut_stage45.png")

# ==========================================================
# LOAD PIPELINE
# ==========================================================
print("Loading Stable Diffusion model...")

pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    safety_checker=None
)

pipe = pipe.to(device)

# Memory optimizations (important for 6GB VRAM)
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

try:
    pipe.enable_xformers_memory_efficient_attention()
    print("xformers enabled.")
except:
    print("xformers not available — continuing safely.")

# ==========================================================
# LOAD LORA
# ==========================================================
print("Loading Clay Animation LoRA...")
pipe.load_lora_weights(LORA_DIR, weight_name=LORA_NAME)

# LoRA strength (0.75–0.85 ideal)
pipe.fuse_lora(lora_scale=0.8)

print("Clay LoRA loaded successfully.")

# ==========================================================
# PROMPT
# ==========================================================
prompt = (
    "clay animation style, handmade plasticine cat character, "
    "anthropomorphic white and gray cat, large round sunglasses, "
    "wearing orange clay scarf and dark jacket, "
    "big expressive amber eyes, small rounded muzzle, "
    "soft sculpted clay fur texture, visible handmade details, "
    "studio portrait, centered composition, clean red background, "
    "soft diffused lighting, stop motion animation frame"
)
negative_prompt = (
    "realistic fur, photorealistic, ultra detailed skin pores, "
    "anime style, manga, low resolution, blurry, noisy, "
    "harsh shadows, dramatic cinematic lighting"
)
# ==========================================================
# GENERATE IMAGE
# ==========================================================
print("Generating Peanut Stage-1 image...")

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=28,
    guidance_scale=6.5,
    width=512,
    height=512
).images[0]

# ==========================================================
# SAVE IMAGE
# ==========================================================
image.save(OUTPUT_PATH)

print("✅ Stage-1 Peanut image saved at:", OUTPUT_PATH)