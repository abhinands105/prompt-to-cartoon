import os
import shutil

# -----------------------------
# PATHS
# -----------------------------
DATA_DIR = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\data"
EXTRA_DIR = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\extra_files"

os.makedirs(EXTRA_DIR, exist_ok=True)

# -----------------------------
# STYLE FOLDERS (KEEP THESE)
# -----------------------------
KEEP_FOLDERS = {
    "adventure_time",
    "alice_forever_after",
    "avatar_-_the_last_airbender",
    "bugs_bunny",
    "dexter’s_laboratory",
    "donald_marvel",
    "garfield_minus_garfield",
    "looney_tunes_comic_book_style",
    "over_the_garden_wall",
    "peanut",
    "peanut_butter_monkey_and_jelly_girl",
    "samurai_jack",
}

# -----------------------------
# MOVE OTHERS
# -----------------------------
for folder in os.listdir(DATA_DIR):

    folder_path = os.path.join(DATA_DIR, folder)

    if not os.path.isdir(folder_path):
        continue

    if folder not in KEEP_FOLDERS:
        print(f"➡ Moving: {folder}")
        shutil.move(folder_path, os.path.join(EXTRA_DIR, folder))

print("✅ Cleanup complete!")