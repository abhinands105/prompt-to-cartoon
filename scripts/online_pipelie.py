import os
import torch
import cv2
from PIL import Image

from huggingface_hub import list_repo_files, hf_hub_download

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler
)

# =====================================================
# DEVICE
# =====================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# =====================================================
# HUGGINGFACE REPO
# =====================================================

REPO_ID = "abhinand105/cartoon-lora-collection"

# =====================================================
# BASE MODEL
# =====================================================

BASE_MODEL = "Lykon/dreamshaper-8"

# =====================================================
# CONTROLNET
# =====================================================

CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15_lineart"

# =====================================================
# OUTPUT
# =====================================================

OUTPUT_DIR = "lora_tests"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# LOAD BASE PIPELINE
# =====================================================

print("\nLoading Base Model...")

pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    safety_checker=None
)

pipe.scheduler = DPMSolverMultistepScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear"
)

pipe = pipe.to(device)

pipe.enable_attention_slicing()
pipe.enable_model_cpu_offload()

# =====================================================
# LOAD CONTROLNET
# =====================================================

print("\nLoading ControlNet...")

controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_MODEL,
    torch_dtype=torch.float16
)

control_pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None
)

control_pipe.scheduler = UniPCMultistepScheduler.from_config(
    control_pipe.scheduler.config
)

control_pipe = control_pipe.to(device)

control_pipe.enable_attention_slicing()
control_pipe.enable_model_cpu_offload()

# =====================================================
# GET ALL LORA FILES
# =====================================================

print("\nFetching LoRA list...")

repo_files = list_repo_files(REPO_ID)

lora_files = [f for f in repo_files if f.endswith(".safetensors")]

print("Found LoRAs:", len(lora_files))

# =====================================================
# USER PROMPT
# =====================================================

prompt = input("\nEnter test prompt: ")

negative_prompt = (
    "text, speech bubble, caption, watermark, logo, "
    "comic panel, letters, words"
)

# =====================================================
# LOOP THROUGH ALL LORAS
# =====================================================

for lora_file in lora_files:

    print("\n==============================")
    print("Testing LoRA:", lora_file)

    # download LoRA
    lora_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=lora_file
    )

    lora_name = os.path.splitext(os.path.basename(lora_file))[0]

    save_dir = os.path.join(OUTPUT_DIR, lora_name)
    os.makedirs(save_dir, exist_ok=True)

    # unload previous LoRA
    try:
        pipe.unload_lora_weights()
        control_pipe.unload_lora_weights()
    except:
        pass

    # =================================================
    # LOAD LORA
    # =================================================

    pipe.load_lora_weights(lora_path)
    pipe.fuse_lora(lora_scale=0.8)

    control_pipe.load_lora_weights(lora_path)
    control_pipe.fuse_lora(lora_scale=1.0)

    generator = torch.Generator(device=device).manual_seed(123)

    # =================================================
    # GENERATE BASE IMAGE
    # =================================================

    print("Generating base image...")

    base = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=512,
        height=512,
        num_inference_steps=30,
        guidance_scale=7,
        generator=generator
    ).images[0]

    base_path = os.path.join(save_dir, "base.png")
    base.save(base_path)

    # =================================================
    # LINEART EXTRACTION
    # =================================================

    img = cv2.imread(base_path)
    img = cv2.resize(img, (512, 512))

    edges = cv2.Canny(img, 150, 250)

    lineart_img = Image.fromarray(edges)

    init_image = Image.open(base_path)

    # =================================================
    # CONTROLNET REFINEMENT
    # =================================================

    print("Refining with ControlNet...")

    final = control_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        control_image=lineart_img,
        strength=0.18,
        num_inference_steps=16,
        guidance_scale=6,
        generator=generator
    ).images[0]

    final_path = os.path.join(save_dir, "final.png")
    final.save(final_path)

    print("Saved:", final_path)

print("\nAll LoRA models tested!")