import os
import torch
import cv2
import numpy as np
from PIL import Image

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel
)

# ----------------------------
# DEVICE
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ----------------------------
# MODELS
# ----------------------------
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
CONTROLNET_MODEL = "lllyasviel/sd-controlnet-canny"

LORA_DIR = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\dataset\advtime_lora_captioned_v6"
LORA_NAME = "advtime_captioned_v6.safetensors"

REF_IMAGE = os.path.join("outputs", "stage1", "stage1_rawdata.png")
OUT_DIR = os.path.join("outputs", "video_frames")
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# LOAD CONTROLNET
# ----------------------------
controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_MODEL,
    torch_dtype=torch.float16
)

# ----------------------------
# LOAD PIPELINE
# ----------------------------
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)

pipe.enable_attention_slicing()

# ----------------------------
# LOAD LORA
# ----------------------------
pipe.load_lora_weights(LORA_DIR, weight_name=LORA_NAME)
print("✅ LoRA loaded successfully (Stage-3)")

# ----------------------------
# LOAD REFERENCE IMAGE
# ----------------------------
if not os.path.exists(REF_IMAGE):
    raise FileNotFoundError("❌ Stage-1 reference image not found")

ref_img = Image.open(REF_IMAGE).convert("RGB")

# ----------------------------
# VIDEO-OPTIMIZED PROMPT
# ----------------------------
prompt = (
    "advtime_scene, "
    "small cartoon boy and a yellow stretchy dog walking together, "
    "side view composition, "
    "simple grassy outdoor path with soft hills, "
    "clear sky background, "
    "adventure time style, "
    "flat pastel colors, "
    "thick clean black outlines, "
    "simple rounded shapes, "
    "2d western cartoon illustration"
)

negative_prompt = (
    "realistic, photorealistic, anime, manga, chibi, "
    "3d render, cinematic lighting, shadows, "
    "thin lines, detailed textures, realistic anatomy, "
    "watermark, logo"
)

# ----------------------------
# GENERATE VIDEO FRAMES
# ----------------------------
NUM_FRAMES = 30   # 2.5 seconds @ 12 fps (smooth clip)

for i in range(NUM_FRAMES):
    np_img = np.array(ref_img)

    # ✅ SMOOTH EDGE JITTER (ANTI-FLICKER)
    edges = cv2.Canny(
        np_img,
        100 + (i % 2),
        200 + (i % 2)
    )

    edges = Image.fromarray(edges)

    frame = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=edges,
        num_inference_steps=28,
        guidance_scale=6.5,
        controlnet_conditioning_scale=1.0
    ).images[0]

    frame_path = os.path.join(OUT_DIR, f"frame_{i:03d}.png")
    frame.save(frame_path)
    print("Saved:", frame_path)

print("✅ VIDEO FRAMES GENERATED SUCCESSFULLY")
