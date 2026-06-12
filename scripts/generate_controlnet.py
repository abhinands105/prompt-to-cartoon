import os
import torch
import cv2
import numpy as np
from PIL import Image

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler

# ==========================================================
# OFFLINE MODE
# ==========================================================

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ==========================================================
# PATHS
# ==========================================================

BASE_MODEL = r"C:\AI\models\dreamshaper8"

LORA_PATH = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\lora_output\new setup\advtime_style_v1.safetensors"

lineart_path = r"C:\Users\abhis\OneDrive\Desktop\project s8\models\controlnet\controlnet_lineart"
softedge_path = r"C:\Users\abhis\OneDrive\Desktop\project s8\models\controlnet\controlnet_softedge"
openpose_path = r"C:\Users\abhis\OneDrive\Desktop\project s8\models\controlnet\controlnet_openpose"

input_image = r"C:\Users\abhis\OneDrive\Desktop\project s8\outputs\adv17.png"
output_image = r"C:\Users\abhis\OneDrive\Desktop\project s8\outputs\adv17_refined.png"

# ==========================================================
# LOAD CONTROLNET MODELS
# ==========================================================

print("Loading ControlNet models...")

lineart = ControlNetModel.from_pretrained(
    lineart_path,
    torch_dtype=torch.float16,
    local_files_only=True
)

softedge = ControlNetModel.from_pretrained(
    softedge_path,
    torch_dtype=torch.float16,
    local_files_only=True
)

openpose = ControlNetModel.from_pretrained(
    openpose_path,
    torch_dtype=torch.float16,
    local_files_only=True
)

# ==========================================================
# LOAD PIPELINE
# ==========================================================

print("Loading pipeline...")

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=[openpose, softedge, lineart],
    torch_dtype=torch.float16,
    safety_checker=None,
    local_files_only=True
).to(device)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

pipe.enable_attention_slicing()
pipe.vae.enable_slicing()

print("Pipeline loaded")

# ==========================================================
# LOAD LORA
# ==========================================================

print("Loading LoRA...")

pipe.load_lora_weights(LORA_PATH)
pipe.fuse_lora(lora_scale=1.3)

print("LoRA loaded")

# ==========================================================
# PREPARE CONTROL IMAGES
# ==========================================================

img = cv2.imread(input_image)
img = cv2.resize(img, (768,768))

softedge_img = cv2.Canny(img, 100, 200)
lineart_img = cv2.Canny(img, 150, 250)

pose_img = np.zeros((768,768,3), dtype=np.uint8)

softedge_img = Image.fromarray(softedge_img)
lineart_img = Image.fromarray(lineart_img)
pose_img = Image.fromarray(pose_img)

control_images = [pose_img, softedge_img, lineart_img]

# ==========================================================
# PROMPT
# ==========================================================

prompt = (
    "masterpiece, best quality, cartoon illustration, "
    "(advtime_style:1.35), "
    "finn_char exploring forest with backpack, "
    "jake_char sniffing ground like detective, "
    "fantasy forest environment with bright green trees, "
    "cartoon adventure scene, flat colors, bold outlines"
)
negative_prompt = (
    "speech bubble, speech balloon, dialogue bubble, dialogue balloon, "
    "text bubble, text balloon, word balloon, caption box, caption bubble, "
    "comic text, dialogue text, letters, words, subtitle, font, paragraph, sentence, "
    "comic panel border, comic layout, watermark, logo"
)

# ==========================================================
# GENERATE IMAGE
# ==========================================================

print("Generating refined image...")

generator = torch.Generator(device=device).manual_seed(123)

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=control_images,
    num_inference_steps=30,
    guidance_scale=7,
    generator=generator
).images[0]

# ==========================================================
# SAVE
# ==========================================================

image.save(output_image)

print("Saved:", output_image)