import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import PeftModel  # required for new diffusers versions

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

LORA_PATH = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\lora_output\new setup\advtime_style_v1.safetensors"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "adv17.png")

# ==========================================================
# LOAD BASE MODEL
# ==========================================================
print("Loading base model...")


pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    safety_checker=None,
    local_files_only=True
)

# Fix scheduler compatibility
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

print("Base model loaded")

# ==========================================================
# LOAD LORA
# ==========================================================
print("Loading LoRA...")

pipe.load_lora_weights(LORA_PATH)
pipe.fuse_lora(lora_scale=1.30)

print("LoRA loaded successfully")

# ==========================================================
# PROMPT
# ==========================================================
prompt = (
    "masterpiece, best quality, ultra detailed, "
    "portrait of a beautiful young woman, "
    "abhi_pencil_portrait, "
    "realistic graphite pencil sketch, "
    "soft shading, cross hatching, "
    "highly detailed face, studio lighting"
)

negative_prompt = (
    "color, colored, painting, watercolor, oil paint, "
    "anime, cartoon, comic style, "
    "3d render, cgi, plastic skin, "
    "blurry, low quality, low resolution, "
    "bad anatomy, distorted face, extra fingers, "
    "text, watermark, logo"
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
        guidance_scale=6,
        width=720,
        height=720,
        generator=generator
    ).images[0]

# ==========================================================
# SAVE IMAGE
# ==========================================================
image.save(OUTPUT_PATH)

print("Image saved at:", OUTPUT_PATH)