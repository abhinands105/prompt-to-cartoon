import os
import torch
import re
import cv2
from PIL import Image

from transformers import AutoTokenizer, AutoModelForCausalLM

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
    LCMScheduler
)

# =====================================================
# OFFLINE MODE
# =====================================================

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# =====================================================
# DEVICE
# =====================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =====================================================
# PATHS
# =====================================================

MAGICPROMPT_PATH = r"C:\Users\abhis\OneDrive\Desktop\project s8\models\magicprompt"
BASE_MODEL = r"C:\AI\models\dreamshaper8"

LORA_PATH = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\lora_output\new setup\advtime_style_v1.safetensors"
LINEART_PATH = r"C:\Users\abhis\OneDrive\Desktop\project s8\models\controlnet\controlnet_lineart"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_IMAGE = os.path.join(OUTPUT_DIR, "base.png")
FINAL_IMAGE = os.path.join(OUTPUT_DIR, "final.png")

# =====================================================
# STYLE MAP
# =====================================================

STYLE_MAP = {
    "adventure_time": "advtime_style"
}

CHARACTER_MAP = {
    "adventure_time": "finn_char"
}

# =====================================================
# LOAD MAGIC PROMPT
# =====================================================

print("Loading MagicPrompt...")

tokenizer = AutoTokenizer.from_pretrained(
    MAGICPROMPT_PATH,
    local_files_only=True
)

tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MAGICPROMPT_PATH,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    local_files_only=True
).to(device)

model.eval()

print("MagicPrompt loaded")

# =====================================================
# PROMPT FUNCTIONS
# =====================================================

def clean_prompt(text):
    banned = ["realistic", "4k", "8k", "photorealistic"]
    text = text.lower()
    for b in banned:
        text = text.replace(b, "")
    return text.strip()

def enhance_prompt(user_prompt):
    inputs = tokenizer(user_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_new_tokens=40,
            temperature=0.8
        )

    return clean_prompt(tokenizer.decode(output[0], skip_special_tokens=True))

def build_prompt(user_prompt):
    enhanced = enhance_prompt(user_prompt)

    return (
        "masterpiece, best quality, "
        "(advtime_style:1.2), finn_char, "
        "flat cartoon style, clean outlines, "
        + enhanced
    )

# =====================================================
# USER INPUT
# =====================================================

user_prompt = input("\nEnter prompt: ")
prompt = build_prompt(user_prompt)

negative_prompt = "text, watermark, realistic, bad anatomy"

# =====================================================
# LOAD BASE PIPELINE (LCM)
# =====================================================

pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    safety_checker=None,
    local_files_only=True
).to(device)

 
# ✅ LCM
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# ✅ YOUR LoRA
pipe.load_lora_weights(
    os.path.dirname(LORA_PATH),
    weight_name=os.path.basename(LORA_PATH)
)
pipe.fuse_lora(lora_scale=0.8)

pipe.enable_attention_slicing()
pipe.enable_model_cpu_offload()

generator = torch.Generator(device=device).manual_seed(42)

# =====================================================
# GENERATE BASE IMAGE
# =====================================================

print("\nGenerating base image...")

base = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=8,
    guidance_scale=0.4,
    width=768,
    height=512,
    generator=generator
).images[0]

base.save(BASE_IMAGE)
print("Base saved")

# =====================================================
# LOAD CONTROLNET
# =====================================================

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

 
# ✅ LCM
control_pipe.scheduler = LCMScheduler.from_config(control_pipe.scheduler.config)

control_pipe.load_lora_weights(
    os.path.dirname(LORA_PATH),
    weight_name=os.path.basename(LORA_PATH)
)

control_pipe.fuse_lora(lora_scale=1.0)

control_pipe.enable_attention_slicing()
control_pipe.enable_model_cpu_offload()

# =====================================================
# EDGE DETECTION
# =====================================================

img = cv2.imread(BASE_IMAGE)
edges = cv2.Canny(img, 150, 250)

lineart_img = Image.fromarray(edges)
init_image = Image.open(BASE_IMAGE).resize((768, 512))

# =====================================================
# FINAL GENERATION
# =====================================================

print("\nRefining image...")

final = control_pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_image,
    control_image=lineart_img,
    strength=0.3,                # 🔥 changed
    num_inference_steps=12,      # 🔥 changed
    guidance_scale=2,
    generator=generator
).images[0]

final = base
final.save(FINAL_IMAGE)

print("\n✅ DONE! Final saved:", FINAL_IMAGE)