import os
import torch
import cv2
from PIL import Image

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler
)

# ==========================================================
# OFFLINE MODE
# ==========================================================

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ==========================================================
# PATHS
# ==========================================================

BASE_MODEL = r"C:\AI\models\dreamshaper8"

LORA_PATH = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\lora_output\new setup\advtime_style_v1.safetensors"

LINEART_PATH = r"C:\Users\abhis\OneDrive\Desktop\project s8\models\controlnet\controlnet_lineart"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_IMAGE_PATH = os.path.join(OUTPUT_DIR, "adv41_base.png")
FINAL_IMAGE_PATH = os.path.join(OUTPUT_DIR, "adv41_final.png")

# ==========================================================
# PROMPT
# ==========================================================

prompt = (
    "masterpiece, best quality, cartoon illustration, "
    "(advtime_style:1.35), "
    "Adventure Time animation style, cel shading, flat colors, bold outlines, "
    "finn_char, jake_char, and princess_bubblegum standing together on forest path"
)

negative_prompt = (
    "speech bubble, dialogue bubble, text balloon, caption box, "
    "comic text, letters, watermark, logo"
)

# ==========================================================
# STAGE 1 — BASE IMAGE
# ==========================================================

print("Loading base pipeline...")

pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    safety_checker=None,
    local_files_only=True
)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    algorithm_type="dpmsolver++"
)

pipe = pipe.to(device)

pipe.enable_attention_slicing()
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()

print("Loading LoRA...")

pipe.load_lora_weights(LORA_PATH)
pipe.fuse_lora(lora_scale=1.35)

generator = torch.Generator(device=device).manual_seed(123)

print("Generating base image...")

base_image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=20,
    guidance_scale=6,
    width=512,
    height=512,
    generator=generator
).images[0]

base_image.save(BASE_IMAGE_PATH)

print("Base image saved:", BASE_IMAGE_PATH)

# ==========================================================
# STAGE 2 — CONTROLNET OUTLINE REFINEMENT
# ==========================================================

print("Loading Lineart ControlNet...")

lineart = ControlNetModel.from_pretrained(
    LINEART_PATH,
    torch_dtype=torch.float16,
    local_files_only=True
)

print("Loading ControlNet Img2Img pipeline...")

control_pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=lineart,
    torch_dtype=torch.float16,
    safety_checker=None,
    local_files_only=True
).to(device)

control_pipe.scheduler = UniPCMultistepScheduler.from_config(
    control_pipe.scheduler.config
)

control_pipe.enable_attention_slicing()
control_pipe.enable_model_cpu_offload()
control_pipe.vae.enable_slicing()

control_pipe.load_lora_weights(LORA_PATH)
control_pipe.fuse_lora(lora_scale=1.35)

# ==========================================================
# PREPARE LINEART IMAGE
# ==========================================================

img = cv2.imread(BASE_IMAGE_PATH)
img = cv2.resize(img, (512, 512))

lineart_img = cv2.Canny(img, 150, 250)
lineart_img = Image.fromarray(lineart_img)

init_image = Image.open(BASE_IMAGE_PATH).resize((512,512))

# ==========================================================
# FINAL IMAGE (REFINE ONLY)
# ==========================================================

print("Refining outlines (background preserved)...")

final_image = control_pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_image,
    control_image=lineart_img,
    strength=0.25,          # VERY IMPORTANT (keeps background)
    num_inference_steps=14,
    guidance_scale=5.5,
    generator=generator
).images[0]

final_image.save(FINAL_IMAGE_PATH)

print("Final image saved:", FINAL_IMAGE_PATH)