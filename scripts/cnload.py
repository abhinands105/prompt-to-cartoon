import torch
import os
from diffusers import StableDiffusionPipeline

# ==========================================================
# DEVICE
# ==========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ==========================================================
# PATHS
# ==========================================================
BASE_MODEL = "Lykon/dreamshaper-8"

LORA_PATH = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\outputs\advtime_lora\advtime_style_lora-step00008961.safetensors"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "advtime_result5.png")

# ==========================================================
# LOAD BASE MODEL
# ==========================================================
print("Loading DreamShaper 8...")

pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    safety_checker=None
)

# Move to GPU FIRST
pipe = pipe.to(device)

# 🔥 IMPORTANT FOR 6GB GPU
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()
# ❌ DO NOT enable attention slicing (causes scale error with LoRA)

print("Base model loaded ✅")

# ==========================================================
# LOAD LORA
# ==========================================================
print("Loading Adventure Style LoRA...")

pipe.load_lora_weights(LORA_PATH)

# Fuse LoRA (recommended for diffusers 0.24)
pipe.fuse_lora(lora_scale=0.85)

print("LoRA loaded successfully ✅")

# ==========================================================
# PROMPT
# ==========================================================
prompt = (
    "advtime_style, full body cartoon boy hero inside a wooden treehouse, "
    "large round head, small black dot eyes, soft smile, "
    "dark blue t shirt, black shorts, blue sneakers with white soles, "
    "short dark hair styled into two small buns, "
    "light blue cape flowing behind him, "
    "holding a glowing blue sword in both hands, dynamic action stance, "
    
    "small black puppy companion with round body and red collar, "
    "looking up at the hero, simple oval shape, flat black color, "
    
    "wood plank floor, curved wooden beams, large rounded window frame, "
    "warm sunlight streaming inside, soft forest background outside window, "
    
    "thin clean black outlines, flat pastel color palette, "
    "minimal shading, simple 2d western animation style, no texture"
)

negative_prompt = (
    "realistic, photorealistic, anime style, manga style, 3d render, cgi, "
    "hyper detailed skin, cinematic lighting, dramatic shadows, "
    "complex textures, ultra detailed realism"
)
# ==========================================================
# GENERATE
# ==========================================================
print("Generating image...")

with torch.autocast("cuda"):
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