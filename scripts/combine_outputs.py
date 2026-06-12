import os
from PIL import Image

# ----------------------------
# INPUT FOLDERS
# ----------------------------
STAGE1_DIR = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\outputs\stage1"
STAGE2_DIR = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\outputs\stage2"

OUTPUT_PATH = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\outputs\final_comparison.png"

# ----------------------------
# LOAD IMAGES
# ----------------------------
stage1_images = sorted([
    os.path.join(STAGE1_DIR, f)
    for f in os.listdir(STAGE1_DIR)
    if f.endswith(".png")
])

stage2_images = sorted([
    os.path.join(STAGE2_DIR, f)
    for f in os.listdir(STAGE2_DIR)
    if f.endswith(".png")
])

# Keep same count
count = min(len(stage1_images), len(stage2_images))
stage1_images = stage1_images[:count]
stage2_images = stage2_images[:count]

# ----------------------------
# OPEN IMAGES
# ----------------------------
stage1_imgs = [Image.open(img).resize((512, 512)) for img in stage1_images]
stage2_imgs = [Image.open(img).resize((512, 512)) for img in stage2_images]

# ----------------------------
# CREATE CANVAS
# ----------------------------
cols = count
rows = 2

canvas_width = cols * 512
canvas_height = rows * 512

canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

# ----------------------------
# PASTE IMAGES
# ----------------------------
for i in range(cols):
    canvas.paste(stage1_imgs[i], (i * 512, 0))
    canvas.paste(stage2_imgs[i], (i * 512, 512))

# ----------------------------
# SAVE OUTPUT
# ----------------------------
canvas.save(OUTPUT_PATH)
print("✅ Combined image saved at:", OUTPUT_PATH)
