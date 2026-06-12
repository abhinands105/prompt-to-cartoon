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

# ⚠️ change to Avatar LoRA
LORA_PATH = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\lora_output\new setup\advtime_style_v1.safetensors"

LINEART_PATH = r"C:\Users\abhis\OneDrive\Desktop\project s8\models\controlnet\controlnet_lineart"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_IMAGE = os.path.join(OUTPUT_DIR, "AdventureTimeB12.png")
FINAL_IMAGE = os.path.join(OUTPUT_DIR, "AdventureTimeF12.png")

# =====================================================
# STYLE + CHARACTER MAP
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

print("MagicPrompt loaded successfully")

# =====================================================
# PROMPT CLEANER
# =====================================================

def clean_prompt(text):

    banned_words = [
        "artstation","concept art","oil painting",
        "greg rutkowski","magali villeneuve",
        "artgerm","stanley lau","4k","8k",
        "high definition","insanely detailed",
        "hyper detailed","photorealistic","realistic"
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

    inputs = tokenizer(
        user_prompt,
        return_tensors="pt",
        padding=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():

        output = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=40,
            do_sample=True,
            temperature=0.8,
            top_k=80,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )

    enhanced = tokenizer.decode(output[0], skip_special_tokens=True)

    return clean_prompt(enhanced)

# =====================================================
# PROMPT BUILDER
# =====================================================
def build_final_prompt(user_prompt, style):

    enhanced = enhance_prompt(user_prompt)

    style_token = STYLE_MAP[style]
    character_token = CHARACTER_MAP[style]

    if style == "adventure_time":
        style_block = (
            "minimalist flat cartoon style, "
            "clean thin black outlines, "
            "flat vibrant colors, "
            "simple geometric shapes, "
            "cartoon network style, "
            "wide environment composition, "
        )

    elif style == "avatar":
        style_block = (
            "bold black outlines, clean cel shading, "
            "dynamic action pose, dramatic lighting, "
        )

    else:
        style_block = ""

    final_prompt = (
        "masterpiece, best quality, "
        f"({style_token}:1.2), "
        f"{character_token}, "
        "solo character, "
        + style_block +
        enhanced
    )

    return enhanced, final_prompt

# =====================================================
# USER PROMPT
# =====================================================

user_prompt = input("\nEnter your prompt: ")

enhanced, prompt = build_final_prompt(user_prompt, "adventure_time")

print("\nEnhanced Prompt:", enhanced)
print("\nFinal Prompt:", prompt)

# =====================================================
# NEGATIVE PROMPT
# =====================================================

negative_prompt = (
    "speech bubble, dialogue bubble, text, caption, subtitle, letters, words, "
    "comic panel, panel border, page layout, "
    "watermark, logo, signature, "
    "realistic, photorealistic, detailed shading, "
    "bad anatomy, extra limbs, duplicate character, multiple characters"
)
# =====================================================
# LOAD STABLE DIFFUSION
# =====================================================

pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    safety_checker=None,
    local_files_only=True
).to(device)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    algorithm_type="dpmsolver++"
)

pipe.enable_attention_slicing()
pipe.enable_model_cpu_offload()

# =====================================================
# LOAD LORA
# =====================================================

pipe.load_lora_weights(
    os.path.dirname(LORA_PATH),
    weight_name=os.path.basename(LORA_PATH)
)

pipe.fuse_lora(lora_scale=.85)

generator = torch.Generator(device=device).manual_seed(
    torch.randint(0, 999999, (1,)).item()
)

# =====================================================
# BASE IMAGE GENERATION
# =====================================================

print("\nGenerating base image...")

base = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=24,
    guidance_scale=6,
    width=768,
    height=512,
    generator=generator
).images[0]

base.save(BASE_IMAGE)

print("Base image saved:", BASE_IMAGE)

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

control_pipe.scheduler = UniPCMultistepScheduler.from_config(
    control_pipe.scheduler.config
)

control_pipe.enable_attention_slicing()
control_pipe.enable_model_cpu_offload()

control_pipe.load_lora_weights(
    os.path.dirname(LORA_PATH),
    weight_name=os.path.basename(LORA_PATH)
)

control_pipe.fuse_lora(lora_scale=1.0)

# =====================================================
# EDGE EXTRACTION
# =====================================================

img = cv2.imread(BASE_IMAGE)
img = cv2.resize(img, (768, 512))

edges = cv2.Canny(img, 150, 250)

lineart_img = Image.fromarray(edges)
init_image = Image.open(BASE_IMAGE).resize((768, 512))

# =====================================================
# FINAL REFINEMENT
# =====================================================

print("\nRefining outlines...")

final = control_pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_image,
    control_image=lineart_img,
    strength=0.15,
    num_inference_steps=14,
    guidance_scale=6,
    generator=generator
).images[0]

final.save(FINAL_IMAGE)

print("Final image saved:", FINAL_IMAGE)