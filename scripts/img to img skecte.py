import os
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
    AutoencoderKL
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- PATHS ---
BASE_MODEL_PATH = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\models\base\stable_diffusion_v1_5\Realistic_Vision_V5.1_fp16-no-ema.safetensors"
LORA_FOLDER = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\lora_output\sketes new"
LORA_WEIGHT = "last-step00001500.safetensors"
INPUT_IMAGE_PATH = r"C:\Users\abhis\Downloads\smartlens_image_1773933229026.jpg"
OUTPUT_PATH = "outputs/graphite_master_final.png"

# --- LOAD MODELS ---
controlnet = [
    ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16).to(device),
    ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_softedge", torch_dtype=torch.float16).to(device)
]

pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
    BASE_MODEL_PATH,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)

pipe.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to(device)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++")

# --- LOAD LORA (Optimized for Heavy Texture) ---
pipe.load_lora_weights(LORA_FOLDER, weight_name=LORA_WEIGHT)
pipe.fuse_lora(lora_scale=0.8) # 0.8 is the sweet spot for heavy texture without breaking the face

# --- PREPROCESSING ---
def get_graphite_maps(path):
    init_img = Image.open(path).convert("RGB").resize((512, 512))
    img_np = np.array(init_img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Map 1: Canny (Identity Anchor)
    # Lower thresholds catch more detail (like hair strands and glasses)
    canny = cv2.Canny(gray, 70, 150) 
    canny_img = Image.fromarray(cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB))
    
    # Map 2: SoftEdge (Shading Guide)
    soft = cv2.GaussianBlur(gray, (5, 5), 0)
    soft = cv2.Sobel(soft, cv2.CV_64F, 1, 1, ksize=3)
    soft = np.clip(np.absolute(soft), 0, 255).astype(np.uint8)
    soft_img = Image.fromarray(cv2.cvtColor(soft, cv2.COLOR_GRAY2RGB))
    
    return init_img, [canny_img, soft_img]

input_img, control_maps = get_graphite_maps(INPUT_IMAGE_PATH)

# --- THE "GRAPHITE" PROMPT ---
# Note the use of weights like (word:1.3) to force the style
prompt = (
    "abhi_sketch_style, (detailed graphite pencil sketch:1.4), "
    "visible pencil strokes, messy cross-hatching, heavy lead shading, "
    "charcoal background scribbles, rough paper grain, "
    "hand-drawn portrait, sharp facial features, realistic eyes, "
    "textured clothing, artistic graphite rendering, monochrome"
)

negative_prompt = (
    "photograph, 3d render, plastic, smooth, blurry, color, "
    "digital painting, oil painting, low contrast, clean skin"
)

# --- GENERATION ---
generator = torch.Generator(device=device).manual_seed(101) # Changed seed for different stroke pattern

with torch.autocast(device):
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=input_img,
        control_image=control_maps,
        
        strength=0.70,                      # CRITICAL: 0.70 turns pixels into lead
        controlnet_conditioning_scale=[1.0, 0.4], # 1.0 on Canny locks the identity
        guidance_scale=9.0,                # High guidance forces the "messy" sketch look
        num_inference_steps=45,
        generator=generator
    )

result.images[0].save(OUTPUT_PATH)
print(f"✅ GRAPHITE MASTER SAVED: {OUTPUT_PATH}")