import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import os

# =========================
# 🔥 CONFIG
# =========================

BASE_MODEL = r"C:\AI\models\dreamshaper8"

LORA_CONFIG = {
    "rabbit": {
        "path": r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\lora_output\new setup\super_rabbit_comic_style_v1.safetensors",
        "trigger": "comic style, bold outlines, vibrant colors",
        "scale": 0.8
    },
    "wrestle": {
        "path": r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\lora_output\new setup\wrestle_heist_graphic_novel_style_v1.safetensors",
        "trigger": "graphic novel style, dark shading, dramatic lighting",
        "scale": 0.85
    }
    # ❌ removed "classic" → incompatible
}

BASE_ENHANCER = (
    "masterpiece, best quality, ultra detailed, cinematic lighting, "
    "soft shadows, sharp focus, professional composition"
)

NEGATIVE_PROMPT = (
    "low quality, blurry, bad anatomy, extra limbs, distorted face, "
    "noise, grain, watermark, text"
)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 🧠 PROMPT BUILDER (FIXED)
# =========================

def build_prompt(user_prompt, lora_name):
    # Avoid duplicate style
    return f"{user_prompt}, {BASE_ENHANCER}"

# =========================
# ⚙️ LOAD PIPELINE
# =========================

print("🚀 Loading model...")

pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# 🔥 SPEED + VRAM OPTIMIZATION
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.to(memory_format=torch.channels_last)
pipe.unet.to(memory_format=torch.channels_last)

print("✅ Model loaded!")

# =========================
# ⚡ LOAD LCM (SUPER SPEED)
# =========================

print("⚡ Loading LCM LoRA...")
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipe.fuse_lora(lora_scale=1.0)

print("✅ LCM loaded!")

# =========================
# 🔥 SAFE LORA LOADER
# =========================

def load_lora_safe(pipe, lora_path, scale):

    try:
        pipe.load_lora_weights(lora_path, adapter_name="style")

        # ✅ Combine LCM + Style properly
        pipe.set_adapters(
            ["lcm", "style"],
            adapter_weights=[1.0, scale]
        )

        print("✅ LoRA loaded (LCM + Style combined)")
        return True

    except Exception as e:
        print("\n❌ LoRA FAILED:", lora_path)
        print("Reason:", str(e)[:200])
        return False

# =========================
# 🎨 GENERATE (FAST MODE)
# =========================

def generate(user_prompt, lora_name):
    print(f"\n🎯 Using LoRA: {lora_name}")

    lora_path = LORA_CONFIG[lora_name]["path"]
    scale = LORA_CONFIG[lora_name]["scale"]

    if not load_lora_safe(pipe, lora_path, scale):
        return

    prompt = build_prompt(user_prompt, lora_name)
    print("✨ Prompt:", prompt)

    image = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=6,   # ⚡ FAST
        guidance_scale=2.0,      # ⚡ REQUIRED for LCM
        height=512,
        width=512
    ).images[0]

    filename = f"{lora_name}_{user_prompt.replace(' ', '_')}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    image.save(path)

    print("💾 Saved:", path)

# =========================
# 🧪 TEST LOOP
# =========================

if __name__ == "__main__":
    while True:
        prompt = input("\n📝 Enter prompt (or exit): ")
        if prompt.lower() == "exit":
            break

        print("Available:", list(LORA_CONFIG.keys()))
        lora = input("🎨 Choose LoRA: ").strip().lower()

        if lora not in LORA_CONFIG:
            print("❌ Invalid LoRA")
            continue

        generate(prompt, lora)