import os
import shutil

# -----------------------------
# DATA ROOT
# -----------------------------
DATA_ROOT = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\data"

for folder_name in os.listdir(DATA_ROOT):

    folder_path = os.path.join(DATA_ROOT, folder_name)

    if not os.path.isdir(folder_path):
        continue

    other_path = os.path.join(folder_path, "other")

    if os.path.exists(other_path) and os.path.isdir(other_path):
        print(f"🗑 Removing 'other' folder from: {folder_name}")
        shutil.rmtree(other_path)

print("✅ All 'other' folders removed successfully!")