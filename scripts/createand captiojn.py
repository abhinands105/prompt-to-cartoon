import os
import shutil

# ========== SOURCE FOLDERS ==========
source_folders = {
    "adventure_time": r"C:\Users\abhis\OneDrive\Pictures\dataset\adventure time",
    "alice_forever_after": r"C:\Users\abhis\OneDrive\Pictures\dataset\Alice Forever After",
    "avatar_last_airbender": r"C:\Users\abhis\OneDrive\Pictures\dataset\Avatar - The Last Airbender",
    "bugs_bunny": r"C:\Users\abhis\OneDrive\Pictures\dataset\Bugs Bunny",
    "dexters_laboratory": r"C:\Users\abhis\OneDrive\Pictures\dataset\Dexter’s Laboratory",
    "donald_marvel": r"C:\Users\abhis\OneDrive\Pictures\dataset\donald marvel",
    "garfield_minus_garfield": r"C:\Users\abhis\OneDrive\Pictures\dataset\Garfield Minus Garfield",
    "looney_tunes_retro": r"C:\Users\abhis\OneDrive\Pictures\dataset\Looney Tunes Comic Book Style",
    "over_garden_wall": r"C:\Users\abhis\OneDrive\Pictures\dataset\over the garden wall",
    "peanut_strip": r"C:\Users\abhis\OneDrive\Pictures\dataset\peanut",
    "pb_monkey_jelly": r"C:\Users\abhis\OneDrive\Pictures\dataset\Peanut Butter Monkey and Jelly Girl",
    "samurai_jack": r"C:\Users\abhis\OneDrive\Pictures\dataset\samurai jack",
    "steven_universe": r"C:\Users\abhis\OneDrive\Pictures\dataset\steven universe",
    "powerpuff_girls": r"C:\Users\abhis\OneDrive\Pictures\dataset\The Powerpuff Girls",
    "wrestle_heist": r"C:\Users\abhis\OneDrive\Pictures\dataset\Wrestle Heist",
}

# ========== DESTINATION ==========
destination_root = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\datasetlora"

# ========== STYLE CAPTIONS ==========
style_captions = {
    "adventure_time": "advtime_style, minimalist flat cartoon style, clean thin outlines, flat colors",
    "alice_forever_after": "modern_indie_style, modern indie comic style, stylized shading",
    "avatar_last_airbender": "avatar_style, anime influenced western cartoon, dynamic action lines",
    "bugs_bunny": "golden_age_cartoon, vintage animation style, bold outlines",
    "dexters_laboratory": "cn_90s_style, cartoon network 90s style, thick outlines",
    "donald_marvel": "superhero_comic_style, dynamic comic shading, action composition",
    "garfield_minus_garfield": "newspaper_strip_style, simple line art, flat colors",
    "looney_tunes_retro": "retro_comic_style, classic comic book texture",
    "over_garden_wall": "storybook_vintage_style, muted colors, textured shading",
    "peanut_strip": "classic_strip_style, minimal line art, flat muted colors",
    "pb_monkey_jelly": "indie_cartoon_style, playful design, soft outlines",
    "samurai_jack": "stylized_graphic_action, bold contrast, cinematic framing",
    "steven_universe": "soft_pastel_cartoon, soft gradient colors, rounded shapes",
    "powerpuff_girls": "bold_shape_cartoon, thick outlines, vibrant flat colors",
    "wrestle_heist": "modern_indie_cartoon, stylized characters, dynamic posing",
}

# ========== PROCESS ==========
for series_name, folder_path in source_folders.items():

    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        continue

    print(f"\nProcessing: {series_name}")

    files = []
    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            full_path = os.path.join(folder_path, file)
            size = os.path.getsize(full_path)
            files.append((full_path, size))

    # Sort by largest file size
    files.sort(key=lambda x: x[1], reverse=True)

    # Select max 300
    selected_files = files[:300]

    print(f"Selected {len(selected_files)} images")

    # Create destination folder
    dest_folder = os.path.join(destination_root, series_name)
    os.makedirs(dest_folder, exist_ok=True)

    # Copy and rename
    for idx, (file_path, _) in enumerate(selected_files, start=1):

        new_image_name = f"{series_name}_{idx:04d}.jpg"
        new_image_path = os.path.join(dest_folder, new_image_name)

        shutil.copy2(file_path, new_image_path)

        # Create caption file
        caption_text = style_captions[series_name]
        caption_file = new_image_name.replace(".jpg", ".txt")
        caption_path = os.path.join(dest_folder, caption_file)

        with open(caption_path, "w", encoding="utf-8") as f:
            f.write(caption_text)

print("\n✅ DONE: Dataset prepared for LoRA training")