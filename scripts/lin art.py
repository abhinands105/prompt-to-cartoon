import os
import torch
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    AutoencoderKL
)

# ==========================================================
# DEVICE
# ==========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ==========================================================
# PATHS (FIXED)
# ==========================================================

BASE_MODEL_PATH = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\models\base\stable_diffusion_v1_5\Realistic_Vision_V5.1_fp16-no-ema.safetensors"

# ✅ BEST MODEL (STEP 1500)
LORA_PATH = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\lora_output\sketes new\last-step00001500.safetensors"

# OUTPUT
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
from datetime import datetime
import random

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
rand = random.randint(100, 999)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"pencil_{timestamp}_{rand}.png")

# ==========================================================
# CHECK FILES
# ==========================================================

assert os.path.exists(BASE_MODEL_PATH), "❌ Base model not found"
assert os.path.exists(LORA_PATH), "❌ LoRA not found"

# ==========================================================
# LOAD BASE MODEL (OFFLINE)
# ==========================================================

print("Loading base model...")

pipe = StableDiffusionPipeline.from_single_file(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
    safety_checker=None
)

# ==========================================================
# LOAD VAE (BETTER QUALITY)
# ==========================================================

print("Loading VAE...")

pipe.vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
    torch_dtype=torch.float16
).to(device)

# ==========================================================
# SCHEDULER (BETTER SAMPLING)
# ==========================================================

pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    algorithm_type="dpmsolver++"
)

pipe = pipe.to(device)
print("🚀 USING DEVICE:", next(pipe.unet.parameters()).device)

# ==========================================================
# VRAM OPTIMIZATION (RTX 3050 SAFE)
# ==========================================================

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

try:
    pipe.enable_xformers_memory_efficient_attention()
except:
    print("xformers not installed")

print("Base model ready")

# ==========================================================
# LOAD LORA (FIXED)
# ==========================================================

print("Loading LoRA...")

pipe.load_lora_weights(LORA_PATH)

pipe = pipe.to(device)

# ✅ ADD THIS HERE (VERY IMPORTANT)
pipe.safety_checker = lambda images, clip_input: (images, [False] * len(images))

print("🚀 USING DEVICE:", next(pipe.unet.parameters()).device)

# ==========================================================
# PROMPT (FIXED TOKEN + OPTIMIZED)
# ==========================================================
prompt = (
    "abhi_sketch_style, time traveler portrait, monochrome, "
    "futuristic trench coat, glowing pocket watch, "
    "windswept hair, mysterious expression, "
    "cinematic sci fi shadows, deep graphite texture, "
    "surreal atmosphere, highly detailed pencil strokes, "
    "masterpiece, ultra realistic sketch"
)

negative_prompt = (
    "nsfw, low quality, blurry, anime, cartoon, 3d render, "
    "bad anatomy, extra limbs, watermark, text"
)
# ==========================================================
# GENERATION SETTINGS
# ==========================================================

print("Generating image...")

generator = torch.Generator(device=device).manual_seed(123)

with torch.autocast(device):
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=28,
        guidance_scale=6.5,
        width=512,
        height=768,
        generator=generator
    )

image = result.images[0]

# ==========================================================
# SAVE OUTPUT
# ==========================================================

image.save(OUTPUT_PATH)

print("✅ Image saved at:", OUTPUT_PATH)