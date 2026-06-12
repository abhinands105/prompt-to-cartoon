from PIL import Image
import os

SRC = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\dataset\advtime_lora\20_advtime"
DST = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\dataset\advtime_lora\20_advtime_fixed"

os.makedirs(DST, exist_ok=True)

for f in os.listdir(SRC):
    if not f.lower().endswith(".png"):
        continue

    img = Image.open(os.path.join(SRC, f)).convert("RGB")
    w, h = img.size

    # skip broken images
    if w < 64 or h < 64:
        continue

    img = img.resize((512, 512), Image.BICUBIC)
    img.save(os.path.join(DST, f))
