import os
import torch
import re
import uuid
from datetime import datetime
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# =========================
# DEVICE
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# PATHS
# =========================
BASE_MODEL = r"C:\AI\models\dreamshaper8"

STYLE_LORA = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\lora_output\new setup\abhi_style_mallu_v1_best.safetensors"

CHAR_LORA_MAP = {
    "pinarayi": r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\lora_output\new setup\abhi_pinarayi_cartoon_v1.safetensors",
    "chandy": r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\lora_output\new setup\abhi_oommen_chandy_cartoon_v1.safetensors",
    "suresh": r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\lora_output\new setup\abhi_suresh_gopi_cartoon_v1.safetensors",
    "modi": r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\lora_output\new setup\abhi_modi_cartoon_v1_600steps.safetensors",
    "amit": r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\lora_output\new setup\abhi_amit_shah_mallu_style_v1.safetensors"
}
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# FILE NAME
# =========================
def generate_filename(prompt):
    uid = uuid.uuid4().hex[:6]
    words = re.sub(r'[^a-zA-Z ]', '', prompt).split()[:2]
    tag = "_".join(words) if words else "img"
    return f"{tag}_{uid}.png"

# =========================
# LOAD MODEL
# =========================
print("🔥 DEVICE:", device)

if BASE_MODEL.endswith(".safetensors"):
    pipe = StableDiffusionPipeline.from_single_file(
        BASE_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
else:
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# 🔥 DISABLE NSFW BLACK IMAGE ISSUE
pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
pipe.requires_safety_checker = False
try:
    pipe.enable_xformers_memory_efficient_attention()
except:
    print("⚠️ xformers not installed, skipping...")

    

# =========================
# CHARACTER DETECTION
# =========================
def detect_characters(prompt):
    p = prompt.lower()
    chars = []

    mapping = {
        "pinarayi": ["pinarayi"],
        "chandy": ["chandy", "oommen"],
        "suresh": ["suresh", "gopi"],
        "modi": ["modi"],
        "amit": ["amit", "shah"]
    }

    for key, keywords in mapping.items():
        for word in keywords:
            if word in p:
                chars.append(key)
                break

    # 🔥 remove duplicates but keep order
    chars = list(dict.fromkeys(chars))

    return chars[:5]
# =========================
# LOAD LORA
# =========================
def load_loras(prompt):

    # 🔥 RESET EVERYTHING
    if hasattr(pipe, "unload_lora_weights"):
        pipe.unload_lora_weights()

    if hasattr(pipe, "set_adapters"):
        try:
            pipe.set_adapters([])
        except:
            pass

    # ✅ ALWAYS LOAD STYLE
    pipe.load_lora_weights(STYLE_LORA, adapter_name="style")

    chars = detect_characters(prompt)
    char_count = len(chars)

    # 🔥 SAFE DEFAULT (prevents crash)
    style_w = 0.5
    char_w = 0.7

    if char_count == 1:
        style_w = 0.6
        char_w = 0.85
    elif char_count == 2:
        style_w = 0.7
        char_w = 0.45
    elif char_count >= 3:
        style_w = 0.65
        char_w = 0.35




    adapters = ["style"]
    weights = [style_w]   # 🔥 style base weight

    # ✅ LOAD CHARACTER LORAS ONLY IF PRESENT
    # 🔥 FIXED WEIGHTS (NO CONFLICT)
    for i, char in enumerate(chars):
        if char in CHAR_LORA_MAP:
            path = CHAR_LORA_MAP[char]

            if os.path.exists(path):
                adapter_name = f"char{i}"
                pipe.load_lora_weights(path, adapter_name=adapter_name)

                adapters.append(adapter_name)

                # ✅ KEY FIX
                if char_count == 1:
                    weight = 0.85
                elif char_count == 2:
                    weight = 0.45 if i == 0 else 0.40   # 🔥 DIFFERENT strengths
                else:
                    weight = 0.35 - (i * 0.05)         # 🔥 decreasing
                    weight = max(0.25, weight)

                weights.append(weight)


    # ✅ APPLY ALL
    pipe.set_adapters(adapters, adapter_weights=weights)

    return chars

# =========================
# BUILD PROMPT 🔥
# =========================
def build_prompt(prompt, chars):

    base = "abhi_style, indian political cartoon, clean line art, minimal shading"

    final_prompt = f"{base}, {prompt}"

    if len(chars) == 1:
        final_prompt += ", single person, full body, centered"

    elif len(chars) == 2:
        final_prompt += ", two people, side by side, natural spacing, full body"

    elif len(chars) >= 3:
        final_prompt += ", three people standing, simple composition, slight spacing"

    return final_prompt
# =========================
# GENERATE
# =========================
def generate(prompt):

    prompt = prompt.strip()

    negative = """
    two heads one body, merged face, fused body,
    overlapping people, duplicate person,
    same face, cloned face, identity mix
    """
    seed = torch.randint(0, 999999, (1,)).item()
    generator = torch.Generator(device=device).manual_seed(seed)

    print("🎲 Seed:", seed)

    chars = load_loras(prompt)
    final_prompt = build_prompt(prompt, chars)

    final_prompt += ", each person separate, not touching"
    final_prompt += ", natural spacing between people"
    final_prompt += ", medium shot, characters clearly visible"
    final_prompt += ", different pose for each person"

# ✅ SAFE TOKEN LIMIT
    final_prompt = " ".join(final_prompt.split()[:40])

    print("🎨 Prompt:", final_prompt)

    image = pipe(
        prompt=final_prompt,
        negative_prompt=negative,
        num_inference_steps=28,
        guidance_scale=6.5,
        width=640,
        height=896,
        generator=generator
    ).images[0]

    filename = generate_filename(prompt)
    path = os.path.join(OUTPUT_DIR, filename)
    image.save(path)

    print("✅ Saved:", path)
# =========================
# RUN
# =========================

if __name__ == "__main__":
    user_prompt = input("Enter prompt: ")
    generate(user_prompt)