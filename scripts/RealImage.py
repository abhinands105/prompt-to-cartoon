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
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler
)

# =====================================================
# 🔥 MODE SWITCH (MAIN CONTROL)
# =====================================================
REAL_MODE = True   # ✅ True = REAL | False = CARTOON

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
# STYLE MAP (ONLY USED IN CARTOON MODE)
# =====================================================
STYLE_MAP = {
    "avatar": "avatar_comic_style",
    "adventure_time": "advtime_style",
    "alice": "alicefa_style",
    "samurai_jack": "samurai_jack_style",
    "steven_universe": "su_style"
}

CHARACTER_MAP = {
    "avatar": "aang_char",
    "adventure_time": "finn_char",
    "alice": "alice_char",
    "samurai_jack": "jack_char",
    "steven_universe": "steven_char"
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
# PROMPT CLEANER (FIXED)
# =====================================================
def clean_prompt(text):

    banned_words = [
        "artstation","concept art","oil painting",
        "greg rutkowski","magali villeneuve",
        "artgerm","stanley lau"
    ]

    text = text.lower()

    for word in banned_words:
        text = text.replace(word, "")

    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s+", ", ", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip(" ,")

# =====================================================
# PROMPT ENHANCER
# =====================================================
def enhance_prompt(user_prompt):

    inputs = tokenizer(user_prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7,
            top_k=60,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    enhanced = tokenizer.decode(output[0], skip_special_tokens=True)
    return clean_prompt(enhanced)

# =====================================================
# PROMPT BUILDER (REAL + CARTOON)
# =====================================================
def build_final_prompt(user_prompt, style):

    enhanced = enhance_prompt(user_prompt)

    if REAL_MODE:
        final_prompt = (
            "masterpiece, best quality, "
            f"{enhanced}, "
            "photorealistic, ultra detailed, "
            "natural lighting, realistic skin texture, "
            "cinematic lighting, sharp focus, 4k"
        )

    else:
        style_token = STYLE_MAP[style]
        character_token = CHARACTER_MAP[style]

        final_prompt = (
            f"masterpiece, best quality, "
            f"({style_token}:1.2), "
            f"{character_token}, "
            f"{enhanced}"
        )

    return enhanced, final_prompt

# =====================================================
# USER INPUT
# =====================================================
user_prompt = input("\nEnter prompt: ")

enhanced, prompt = build_final_prompt(user_prompt, "adventure_time")

print("\nEnhanced:", enhanced)
print("\nFinal Prompt:", prompt)

# =====================================================
# NEGATIVE PROMPT (FIXED)
# =====================================================
if REAL_MODE:
    negative_prompt = (
        "cartoon, anime, illustration, drawing, painting, "
        "low quality, blurry, bad anatomy, extra limbs"
    )
else:
    negative_prompt = (
        "realistic, photorealistic, bad anatomy"
    )

# =====================================================
# LOAD BASE PIPELINE
# =====================================================
pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    safety_checker=None,
    local_files_only=True
).to(device)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.enable_attention_slicing()
pipe.enable_model_cpu_offload()

# =====================================================
# LOAD LORA ONLY FOR CARTOON
# =====================================================
if not REAL_MODE:
    pipe.load_lora_weights(
        os.path.dirname(LORA_PATH),
        weight_name=os.path.basename(LORA_PATH)
    )
    pipe.fuse_lora(lora_scale=0.85)

# =====================================================
# GENERATE BASE IMAGE
# =====================================================
generator = torch.Generator(device=device).manual_seed(
    torch.randint(0, 999999, (1,)).item()
)

print("\nGenerating base image...")

base = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30 if REAL_MODE else 24,
    guidance_scale=7.5 if REAL_MODE else 6,
    width=768,
    height=768 if REAL_MODE else 512,
    generator=generator
).images[0]

base.save(BASE_IMAGE)
print("Saved base:", BASE_IMAGE)

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

control_pipe.scheduler = UniPCMultistepScheduler.from_config(control_pipe.scheduler.config)

control_pipe.enable_attention_slicing()
control_pipe.enable_model_cpu_offload()

# LOAD LORA ONLY FOR CARTOON
if not REAL_MODE:
    control_pipe.load_lora_weights(
        os.path.dirname(LORA_PATH),
        weight_name=os.path.basename(LORA_PATH)
    )
    control_pipe.fuse_lora(lora_scale=1.0)

# =====================================================
# EDGE DETECTION
# =====================================================
img = cv2.imread(BASE_IMAGE)
img = cv2.resize(img, (768, 768 if REAL_MODE else 512))

edges = cv2.Canny(img, 100, 200)

lineart_img = Image.fromarray(edges)
init_image = Image.open(BASE_IMAGE).resize((768, 768 if REAL_MODE else 512))

# =====================================================
# FINAL REFINEMENT
# =====================================================
print("\nRefining image...")

final = control_pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_image,
    control_image=lineart_img,
    strength=0.2 if REAL_MODE else 0.15,
    num_inference_steps=20,
    guidance_scale=7.5 if REAL_MODE else 6,
    generator=generator
).images[0]

final.save(FINAL_IMAGE)

print("\n✅ FINAL IMAGE:", FINAL_IMAGE)