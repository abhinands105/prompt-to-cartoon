import os
from PIL import Image

source_folder = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\datasetlora\line art"
target_folder = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\datasetlora\pencil_portrait_sketch_dataset"

os.makedirs(target_folder, exist_ok=True)

image_numbers = [

# your list here

454,456,458,460,462,464,466,469,471,473,476,478,480,
482,484,486,488,490,492,494,496,498,500,

502,504,506,508,510,512,514,516,518,520,522,524,526,
528,530,532,534,536,538,540,542,544,546,548,550,552,
554,556,558,560,562,564,566,568,570,572,574,576,578,
580,582,584,586,588,590,592,594,596,598,600   
]

for num in image_numbers:

    name = f"line_art_portrait_sketch_{num:03d}"

    jpg_path = os.path.join(source_folder, name + ".jpg")

    if not os.path.exists(jpg_path):
        print(f"❌ Missing: {name}")
        continue

    try:
        img = Image.open(jpg_path).convert("RGB")

        png_path = os.path.join(target_folder, name + ".png")
        img.save(png_path)

        txt_path = os.path.join(target_folder, name + ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("abhi_sketch_style")

        print(f"✅ Processed: {name}")

    except Exception as e:
        print(f"⚠️ Error with {name}: {e}")

print("🎉 Transfer Completed")