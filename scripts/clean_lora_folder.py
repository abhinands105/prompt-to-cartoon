import os
import shutil

folder = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\lora_output\new setup"
target = os.path.join(folder, "LORA UNWANTED FILES")

os.makedirs(target, exist_ok=True)

for f in os.listdir(folder):

    path = os.path.join(folder, f)

    if (
        f.endswith(".json")
        or f.endswith(".toml")
        or "-000" in f
    ):
        shutil.move(path, os.path.join(target, f))

print("Cleanup complete.")