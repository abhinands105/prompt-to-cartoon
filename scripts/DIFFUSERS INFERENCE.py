import torch
from diffusers import StableDiffusionPipeline

base_model = "runwayml/stable-diffusion-v1-5"
lora_path = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\models\lora\type1_gag\advtime_gag.safetensors"

pipe = StableDiffusionPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    safety_checker=None
)

pipe.load_lora_weights(lora_path)
pipe.to("cuda")

prompt = (
    "a cartoon character standing in a whimsical fantasy environment, "
    "simple pose, clean shapes, flat pastel colors, "
    "thick clean black outlines, 2d western cartoon style"
)

negative_prompt = (
    "realistic, photorealistic, anime, manga, 3d, render, cgi, "
    "high detail, complex background, text, watermark"
)

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=25,
    guidance_scale=7,
).images[0]

image.save("advtime_final_output.png")
