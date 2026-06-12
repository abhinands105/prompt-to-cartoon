import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

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

# SD 1.5 base model (download locally first)
BASE_MODEL = r"C:\AI\models\sd15"

# Your trained LoRA
LORA_PATH = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\lora_output\linart1\abhi_pencil_portrait_lora.safetensors"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "pencil_test10.png")

# ==========================================================
# LOAD BASE MODEL
# ==========================================================

print("Loading SD 1.5 base model...")

pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    safety_checker=None,
    local_files_only=True
)

# Better scheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    algorithm_type="dpmsolver++"
)

pipe = pipe.to(device)

# ==========================================================
# VRAM OPTIMIZATION (RTX 3050)
# ==========================================================

pipe.enable_attention_slicing()
pipe.vae.enable_slicing()

print("SD 1.5 loaded")

# ==========================================================
# LOAD LORA
# ==========================================================

print("Loading LoRA...")

pipe.load_lora_weights(LORA_PATH)

pipe.fuse_lora(lora_scale=1.35)

print("LoRA loaded successfully")

# ==========================================================
# PROMPT
# ==========================================================

prompt = (
    "masterpiece, ultra detailed, "
    "(abhi_pencil_portrait:1.4), "
    "portrait of an elderly man, "
    "deep wrinkles, long beard, wise expression, "
    "realistic graphite pencil sketch portrait, "
    "fine cross hatching, detailed shading"
)

negative_prompt = (
    "color, painting, anime, cartoon, 3d render, "
    "notebook page, spiral binding, paper border, "
    "blurry, low quality, text, watermark"
)

# ==========================================================
# GENERATE IMAGE
# ==========================================================

print("Generating image...")

generator = torch.Generator(device=device).manual_seed(123)

with torch.autocast(device):
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=28,
        guidance_scale=7,
        width=512,
        height=512,
        generator=generator
    ).images[0]

# ==========================================================
# SAVE IMAGE
# ==========================================================

image.save(OUTPUT_PATH)

print("Image saved at:", OUTPUT_PATH)