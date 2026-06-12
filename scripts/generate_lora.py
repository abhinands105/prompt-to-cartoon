import torch
import os
from diffusers import StableDiffusionPipeline
import os
os.environ["HF_HUB_OFFLINE"] = "1"
# ==========================================================
# DEVICE
# ==========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ==========================================================
# PATHS
# ==========================================================
BASE_MODEL = r"C:\AI\models\dreamshaper8"   # SD 1.5 based model

LORA_PATH = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\lora_output\new setup\advtime_style_v1.safetensors"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "advtime_result_clean5.png")

# ==========================================================
# LOAD BASE MODEL
# ==========================================================
print("Loading base model...")

pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    safety_checker=None
)

pipe = pipe.to(device)

# 🔥 RTX 3050 (6GB) SAFE SETTINGS
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

print("Base model loaded ✅")

# ==========================================================
# LOAD LORA
# ==========================================================
print("Loading LoRA...")

pipe.load_lora_weights(LORA_PATH)
pipe.fuse_lora(lora_scale=0.85)   # adjust 0.7–0.9 if needed

print("LoRA loaded successfully ✅")

# ==========================================================
# PROMPT (77 TOKEN SAFE VERSION)
# ==========================================================
prompt = (
    "advtime_style, full body cartoon boy hero, "
    "round head, dot eyes, soft smile, "
    "dark blue shirt, black shorts, blue sneakers, "
    "short dark hair buns, light blue cape, glowing blue sword, "
    "dynamic action pose, "
    "small black puppy with red collar, "
    "wooden treehouse interior, warm sunlight, forest outside window, "
    "thin black outlines, flat pastel colors, minimal shading, 2d western animation"
)

negative_prompt = (
    "realistic, photorealistic, anime, manga, 3d render, cgi, "
    "hyper detailed skin, dramatic lighting, complex textures"
)

# ==========================================================
# GENERATE
# ==========================================================
print("Generating image...")

with torch.autocast(device):
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=22,
        guidance_scale=6.5,
        width=512,
        height=512
    ).images[0]

image.save(OUTPUT_PATH)

print("✅ Image saved at:", OUTPUT_PATH)