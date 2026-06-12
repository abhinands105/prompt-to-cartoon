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
BASE_MODEL = r"C:\Users\abhis\Downloads\v1-5-pruned-emaonly.safetensors"

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

if device == "cuda":
    pipe = StableDiffusionPipeline.from_single_file(
        BASE_MODEL,
        torch_dtype=torch.float16
    ).to(device)
else:
    pipe = StableDiffusionPipeline.from_single_file(
        BASE_MODEL,
        torch_dtype=torch.float32
    ).to(device)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
try:
    pipe.enable_xformers_memory_efficient_attention()
except:
    print("⚠️ xformers not installed, skipping...")

# =========================
# CHARACTER DETECTION
# =========================
def detect_character(prompt):
    p = prompt.lower()

    if "pinarayi" in p:
        return "pinarayi"
    if "chandy" in p or "oommen" in p:
        return "chandy"
    if "suresh" in p or "gopi" in p:
        return "suresh"
    if "modi" in p:
        return "modi"
    if "amit" in p or "shah" in p:
        return "amit"

    return None

# =========================
# LOAD LORA
# =========================
def load_loras(prompt):

    # 🔥 Reset LoRAs safely
    if hasattr(pipe, "unload_lora_weights"):
        pipe.unload_lora_weights()

    if hasattr(pipe, "set_adapters"):
        try:
            pipe.set_adapters([])
        except:
            pass

    # ✅ Load style
    pipe.load_lora_weights(STYLE_LORA, adapter_name="style")

    char = detect_character(prompt)

    # ✅ Load character
    if char and char in CHAR_LORA_MAP:
        path = CHAR_LORA_MAP[char]

        if os.path.exists(path):
            pipe.load_lora_weights(path, adapter_name="char")
        else:
            print("⚠️ Missing LoRA:", path)

    # ✅ Apply weights
    if char:
        pipe.set_adapters(["style", "char"], adapter_weights=[0.6, 0.8])
    else:
        pipe.set_adapters(["style"], adapter_weights=[0.7])

    return char

# =========================
# BUILD PROMPT 🔥
# =========================
def build_prompt(user_prompt, char):

    # ✅ CORE STYLE (MATCH DATASET EXACTLY)
    base = "abhi_style, indian political cartoon, caricature, satire, simple line art, minimal shading"

    # ✅ CHARACTER TOKEN
    CHAR_TOKEN_MAP = {
    "modi": "modi_character",
    "amit": "amit_shah_character",
    "chandy": "oommen_chandy_character",
    "pinarayi": "pinarayi_vijayan_character",
    "suresh": "suresh_gopi_character"
}

    if char:
        base = f"{CHAR_TOKEN_MAP[char]}, " + base

    # ✅ EXTRACT SCENE KEYWORDS (VERY IMPORTANT)
    scene = user_prompt.lower()

    # 🔥 SCENE ENHANCEMENT RULES
    scene_tokens = []

    if "talk" in scene or "discussion" in scene:
        scene_tokens.append("two politicians talking, conversation scene")

    if "fight" in scene or "argue" in scene:
        scene_tokens.append("argument scene, pointing gesture")

    if "speech" in scene or "stage" in scene:
        scene_tokens.append("politician speaking at podium, public event")

    if "crowd" in scene:
        scene_tokens.append("crowd background")

    if "police" in scene:
        scene_tokens.append("police officer present")

    if "document" in scene:
        scene_tokens.append("document in hand")

    if "walking" in scene:
        scene_tokens.append("walking pose")

    if "sitting" in scene:
        scene_tokens.append("sitting pose")

    if "angry" in scene:
        scene_tokens.append("angry expression")

    if "happy" in scene:
        scene_tokens.append("smiling expression")

    # 🔥 LIMIT TOKENS
    user_prompt = " ".join(user_prompt.split()[:12])

    # ✅ FINAL PROMPT
    scene_part = ", ".join(scene_tokens) if scene_tokens else ""
    parts = [base]

    if scene_part:
        parts.append(scene_part)

    if user_prompt:
        parts.append(user_prompt)

    return ", ".join(parts)
# =========================
# GENERATE
# =========================
def generate(prompt):

    prompt = " ".join(prompt.split()[:60])
    negative = """
    photorealistic, realistic, 3d render, anime, smooth shading,
    high detail, ultra realistic, cinematic lighting,
    perfect anatomy, detailed skin, soft lighting,
    blurry, low quality, extra fingers, bad anatomy
    """
    seed = torch.randint(0, 999999, (1,)).item()
    generator = torch.Generator(device=device).manual_seed(seed)

    print("🎲 Seed:", seed)
    char = load_loras(prompt)

    # ✅ FIRST build prompt
    final_prompt = build_prompt(prompt, char)

    # ✅ THEN limit tokens
    final_prompt = " ".join(final_prompt.split()[:60])

    print("🎨 Prompt:", final_prompt)

    image = pipe(
        prompt=final_prompt,
        negative_prompt=negative,
        num_inference_steps=30,
        guidance_scale=6.5,
        width=768,
        height=768,
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