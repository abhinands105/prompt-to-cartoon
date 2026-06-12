import os

# 📁 Images directory
IMAGE_DIR = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\dataset\3_Modern_Western_Cartoons\Adventure_Time\images"

# 📝 Base caption (LoRA-friendly)
BASE_CAPTION = (
    "adventure time cartoon style, "
    "animated tv show frame, "
    "bold black outlines, "
    "flat colorful shading, "
    "2d animation, "
    "comic scene"
)

count = 0

for file in os.listdir(IMAGE_DIR):
    if file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        txt_name = file.rsplit(".", 1)[0] + ".txt"
        txt_path = os.path.join(IMAGE_DIR, txt_name)

        # Skip if caption already exists
        if os.path.exists(txt_path):
            continue

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(BASE_CAPTION)

        count += 1

print(f"✅ {count} caption (.txt) files created successfully")
