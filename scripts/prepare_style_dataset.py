import os
import shutil

# -----------------------------
# SOURCE DATASET ROOT
# -----------------------------
SOURCE_ROOT = r"C:\Users\abhis\OneDrive\Pictures\dataset"

# -----------------------------
# DESTINATION ROOT
# -----------------------------
DEST_ROOT = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\data"

os.makedirs(DEST_ROOT, exist_ok=True)

# -----------------------------
# PROCESS EACH FOLDER
# -----------------------------
for folder_name in os.listdir(SOURCE_ROOT):

    source_folder = os.path.join(SOURCE_ROOT, folder_name)

    if not os.path.isdir(source_folder):
        continue

    # Clean folder name for style token
    clean_name = folder_name.lower().replace(" ", "_")
    style_token = f"{clean_name}_style"

    dest_folder = os.path.join(DEST_ROOT, clean_name)
    os.makedirs(dest_folder, exist_ok=True)

    print(f"\n📂 Processing: {folder_name}")

    for file in os.listdir(source_folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):

            src_img_path = os.path.join(source_folder, file)
            dst_img_path = os.path.join(dest_folder, file)

            # Copy image
            shutil.copy2(src_img_path, dst_img_path)

            # Create caption file
            txt_name = os.path.splitext(file)[0] + ".txt"
            txt_path = os.path.join(dest_folder, txt_name)

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(style_token)

    print(f"✅ Done: {folder_name}")

print("\n🎉 ALL DATASETS PREPARED SUCCESSFULLY!")