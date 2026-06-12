import os

PROJECT_ROOT = "project_s8"

folders = [
    "env",
    "data/type1_gag_strips/images",
    "data/type1_gag_strips/captions",
    "data/type2_traditional_classic/images",
    "data/type2_traditional_classic/captions",
    "data/type3_modern_western/images",
    "data/type3_modern_western/captions",
    "data/type4_indie_comics/images",
    "data/type4_indie_comics/captions",
    "training/training_logs",
    "models/base/stable_diffusion_v1_5",
    "models/controlnet/canny",
    "models/controlnet/lineart",
    "models/lora/type1_gag",
    "models/lora/type2_classic",
    "models/lora/type3_modern",
    "models/lora/type4_indie",
    "inference",
    "inputs/control_images",
    "outputs/baseline",
    "outputs/controlnet",
    "outputs/lora",
    "notebooks",
    "docs",
    "utils"
]

files = [
    "README.md",
    "requirements.txt",
    "training/train_lora.py",
    "training/config_lora.yaml",
    "inference/generate_sd_baseline.py",
    "inference/generate_sd_controlnet.py",
    "inference/generate_sd_controlnet_lora.py",
    "docs/project_journal.md",
    "docs/interim_report_phase2.md",
    "utils/image_preprocessing.py",
    "utils/caption_helpers.py",
    "utils/controlnet_utils.py"
]

# Create folders
for folder in folders:
    path = os.path.join(PROJECT_ROOT, folder)
    os.makedirs(path, exist_ok=True)

# Create empty files
for file in files:
    path = os.path.join(PROJECT_ROOT, file)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("")

print("✅ Project structure created successfully!")
