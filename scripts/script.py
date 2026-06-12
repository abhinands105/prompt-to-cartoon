import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")


lora_path = r"C:\Users\abhis\OneDrive\Desktop\project s8\project_s8\project_s8\models\lora\type1_gag\advtime_gag.safetensors"

pipe.load_lora_weights(lora_path)

prompt = "a cartoon character standing in a whimsical fantasy environment,simple pose,adventure time style,flat pastel colors,thick clean black outlines,simple shapes,2d western cartoon style"



image = pipe(
    prompt,
    num_inference_steps=30,
    guidance_scale=6.5
).images[0]

image.save("advtime_repl_test.png")
print("✅ Image saved: advtime_repl_test.png")
