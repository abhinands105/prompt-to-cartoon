import requests
import os

HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-2-1"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
}

def generate_cartoon(prompt, style):

    full_prompt = f"{style} cartoon style, {prompt}"

    payload = {
        "inputs": full_prompt
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(response.text)

    with open("result.png", "wb") as f:
        f.write(response.content)

    return "result.png"