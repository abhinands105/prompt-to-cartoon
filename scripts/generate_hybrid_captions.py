import os
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm

# ----------------------------
# CONFIG
# ----------------------------
IMAGE_DIR = r"dataset/3_Modern_Western_Cartoons/Adventure_Time/images"

STYLE_TOKENS = (
    "adventure time cartoon style, western cartoon, 2d animation, "
    "bold black outlines, flat pastel colors, fantasy cartoon world, clean line art"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# LOAD BLIP
# ----------------------------
print("🔹 Loading BLIP model...")
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(DEVICE)

# ----------------------------
# PROCESS IMAGES
# ----------------------------
image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

print(f"📸 Found {len(image_files)} images")

for img_name in tqdm(image_files, desc="📝 Generating captions"):
    img_path = os.path.join(IMAGE_DIR, img_name)
    txt_path = os.path.splitext(img_path)[0] + ".txt"

    # Skip if caption already exists
    if os.path.exists(txt_path):
        continue

    image = Image.open(img_path).convert("RGB")

    inputs = processor(image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=40)

    base_caption = processor.decode(
        out[0], skip_special_tokens=True
    ).strip()

    final_caption = f"{base_caption}, {STYLE_TOKENS}"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(final_caption)

print("✅ HYBRID captioning completed successfully")
