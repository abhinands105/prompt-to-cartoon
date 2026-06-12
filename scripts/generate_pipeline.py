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

from magicprompt import build_final_prompt

# ===============================
# OFFLINE MODE
# ===============================

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# PATHS
# ===============================

BASE_MODEL = r"C:\AI\models\dreamshaper8"

LORA_PATH = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\lora_output\new setup\alicefa_comic_v1.safetensors"

LINEART_PATH = r"C:\Users\abhis\OneDrive\Desktop\project s8\models\controlnet\controlnet_lineart"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_IMAGE = os.path.join(OUTPUT_DIR, "base1132.png")
FINAL_IMAGE = os.path.join(OUTPUT_DIR, "final1132.png")

# ===============================
# USER PROMPT
# ===============================

user_prompt = "alice  exploring magical forest"

enhanced, prompt = build_final_prompt(user_prompt)

print("\nEnhanced Prompt:", enhanced)
print("\nFinal Prompt:", prompt)

negative_prompt = (
    "speech bubble, dialogue bubble, text balloon, caption box, "
    "comic text, letters, watermark, logo"
)

# ===============================
# BASE PIPELINE
# ===============================

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

pipe.load_lora_weights(LORA_PATH)
pipe.fuse_lora(lora_scale=1.6)

generator = torch.Generator(device=device).manual_seed(123)

print("\nGenerating base image...")

base = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=6.5,
    width=512,
    height=512,
    generator=generator
).images[0]

base.save(BASE_IMAGE)

print("Base image saved:", BASE_IMAGE)

# ===============================
# CONTROLNET
# ===============================

lineart = ControlNetModel.from_pretrained(
    LINEART_PATH,
    torch_dtype=torch.float16,
    local_files_only=True
)

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

control_pipe.load_lora_weights(LORA_PATH)
control_pipe.fuse_lora(lora_scale=1.35)

# ===============================
# LINEART EXTRACTION
# ===============================

img = cv2.imread(BASE_IMAGE)
img = cv2.resize(img, (512, 512))

edges = cv2.Canny(img, 150, 250)
lineart_img = Image.fromarray(edges)

init_image = Image.open(BASE_IMAGE).resize((512,512))

# ===============================
# FINAL IMAGE
# ===============================

print("\nRefining outlines...")

final = control_pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_image,
    control_image=lineart_img,
    strength=0.25,
    num_inference_steps=16,
    guidance_scale=6,
    generator=generator
).images[0]

final.save(FINAL_IMAGE)

print("Final image saved:", FINAL_IMAGE)  