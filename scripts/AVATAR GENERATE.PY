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
# IMPORTANT: Use SAME base model as training
# ==========================================================
BASE_MODEL = r"C:\AI\models\dreamshaper8"   # must be full diffusers folder
LORA_PATH = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\lora_output\new setup\avatar_last_airbender_comic_r16_e10_v1.safetensors"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "avatar_test5290.png")

if not os.path.exists(BASE_MODEL):
    raise FileNotFoundError("Base model folder not found.")

if not os.path.exists(LORA_PATH):
    raise FileNotFoundError("LoRA file not found.")

# ==========================================================
# LOAD MODEL
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
pipe.fuse_lora(lora_scale=0.85)  # start lower than 0.95

# ==========================================================
# TEST PROMPT (Avatar)
# ==========================================================
prompt = (
    "cartoon_style, three school friends sitting together on wooden classroom bench, "
    "teenage students talking and smiling, classroom interior with desks and chairs, "
    "large window showing orange sunset sky outside, warm sunset light entering classroom, "
    "one boy standing near the side of the bench looking toward friends, "
    "soft blue evening sky visible through window, peaceful school atmosphere, "
    "clean cartoon illustration, bold outlines, flat cel shading"
)

negative_prompt = (
    "realistic, photorealistic, 3d render, blurry, low quality"
)
generator = torch.Generator(device).manual_seed(42)

with torch.autocast("cuda"):
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=28,
        guidance_scale=7,
        width=512,
        height=512
    ).images[0]

image.save(OUTPUT_PATH)
print("✅ Image saved at:", OUTPUT_PATH)