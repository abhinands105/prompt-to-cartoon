import os
import torch
from diffusers import StableDiffusionPipeline

# ==========================================================
# OFFLINE MODE
# ==========================================================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ==========================================================
# SAME BASE MODEL USED FOR TRAINING
# ==========================================================
BASE_MODEL = r"C:\AI\models\dreamshaper8"

# YOUR NEW LORA
LORA_PATH = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\lora_output\new setup\bugs_bunny_lt_comic_r32_e8_v1.safetensors"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "bugs_testhd310.png")

# ==========================================================
# LOAD PIPELINE
# ==========================================================
pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    safety_checker=None,
    local_files_only=True
).to(device)

pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

# ==========================================================
# LOAD LORA
# ==========================================================
pipe.load_lora_weights(LORA_PATH)
pipe.fuse_lora(lora_scale=0.9)

# ==========================================================
# BUGS BUNNY PROMPT
# ==========================================================
prompt = (
    "avatar_comic_style, young boy relaxing on a folding beach chair, "
    "holding a lime drink with straw and small umbrella, "
    "sandy beach with gentle ocean waves, colorful beach umbrella nearby, "
    "palm trees and seashells scattered in sand, bright blue sky, warm sunlight, "
    "cel shaded illustration, flat colors, thick clean black outlines"
)

negative_prompt = (
    "photorealistic, 3d render, soft shading, smooth gradients, blurry, low quality"
)
generator = torch.Generator(device).manual_seed(42)

with torch.autocast("cuda"):
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=28,
        guidance_scale=7,
        width=720,
        height=720
    ).images[0]

image.save(OUTPUT_PATH)

print("✅ Image saved at:", OUTPUT_PATH)