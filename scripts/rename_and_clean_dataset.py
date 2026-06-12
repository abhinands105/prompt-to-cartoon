import os
import shutil

# -----------------------------
# ROOT DATA FOLDER
# -----------------------------
DATA_ROOT = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\data"

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

for folder_name in os.listdir(DATA_ROOT):

    folder_path = os.path.join(DATA_ROOT, folder_name)

    if not os.path.isdir(folder_path):
        continue

    print(f"\n📂 Processing: {folder_name}")

    clean_name = folder_name.lower().replace(" ", "_")
    counter = 1

    other_folder = os.path.join(folder_path, "other")
    os.makedirs(other_folder, exist_ok=True)

    files = sorted(os.listdir(folder_path))

    for file in files:

        file_path = os.path.join(folder_path, file)

        # Skip folder itself
        if os.path.isdir(file_path):
            continue

        name, ext = os.path.splitext(file)

        # -----------------------------
        # HANDLE IMAGE FILES
        # -----------------------------
        if ext.lower() in IMAGE_EXTENSIONS:

            new_image_name = f"{clean_name}_{counter:04d}{ext}"
            new_image_path = os.path.join(folder_path, new_image_name)

            # Rename image
            os.rename(file_path, new_image_path)

            # Rename corresponding txt if exists
            old_txt = os.path.join(folder_path, name + ".txt")
            if os.path.exists(old_txt):
                new_txt = os.path.join(folder_path, f"{clean_name}_{counter:04d}.txt")
                os.rename(old_txt, new_txt)

            counter += 1

        # -----------------------------
        # MOVE OTHER FILES
        # -----------------------------
        elif ext.lower() != ".txt":
            shutil.move(file_path, os.path.join(other_folder, file))

    print(f"✅ Done: {counter-1} images renamed")

print("\n🎉 Dataset cleaned and renamed successfully!")