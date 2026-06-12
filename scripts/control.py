import os
import torch
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel
)
from PIL import Image

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
CONTROLNET_MODEL = r"C:\AI\models\controlnet_lineart"
LORA_PATH = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\lora_output\new setup\dm_marvel_style_v1.safetensors"

INPUT_IMAGE = r"outputs\dm.marvel1885fkd3.png"
OUTPUT_IMAGE = r"outputs\refined_output.png"

# ==========================================================
# LOAD CONTROLNET
# ==========================================================
print("Loading ControlNet...")

controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_MODEL,
    torch_dtype=torch.float16,
    local_files_only=True
)

# ==========================================================
# LOAD PIPELINE
# ==========================================================
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None,
    local_files_only=True
).to(device)

pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

print("Pipeline ready ✅")

# ==========================================================
# LOAD LORA
# ==========================================================
pipe.load_lora_weights(LORA_PATH)
pipe.fuse_lora(lora_scale=1.25)

print("LoRA loaded ✅")

# ==========================================================
# LOAD IMAGE
# ==========================================================
image = Image.open(INPUT_IMAGE).convert("RGB").resize((512,512))

# ==========================================================
# PROMPTS
# ==========================================================
prompt = (
    "dm_marvel_style, comic book illustration, "
    "loki inspired cartoon duck sitting on giant golden throne in nordic palace, "
    "large stack of warrior cartoon ducks forming tower toward the throne, "
    "winged hero duck flying above, "
    "grand golden architecture, dramatic low angle view, "
    "bold comic outlines, flat cel shading"
)

negative_prompt = (
    "realistic, photorealistic, anime, manga, 3d render, cgi, "
    "blurry, noisy"
)

# ==========================================================
# REFINE IMAGE
# ==========================================================
print("Refining image...")

generator = torch.Generator(device).manual_seed(42)

with torch.autocast("cuda"):
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        control_image=image,
        strength=0.35,  # keep structure but refine
        num_inference_steps=30,
        guidance_scale=6.5,
        generator=generator
    ).images[0]

result.save(OUTPUT_IMAGE)

print("✅ Refined image saved:", OUTPUT_IMAGE)