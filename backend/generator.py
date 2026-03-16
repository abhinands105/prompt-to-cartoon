import requests
import os

HF_TOKEN = os.getenv("HF_TOKEN")
# Target the specific model endpoint directly
API_URL = "https://api-inference.huggingface.co/models/Lykon/DreamShaper"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_cartoon(prompt, style):
    # Constructing the prompt with the style prefix
    full_prompt = f"{style} cartoon style, {prompt}, high quality, vector art"

    payload = {
        "inputs": full_prompt,
        "parameters": {
            "negative_prompt": "realistic, photo, grainy, low resolution, blurry",
            "wait_for_model": True
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        print(f"HF ERROR ({response.status_code}): {response.text}")
        raise Exception(response.text)

    image_path = os.path.join(OUTPUT_DIR, "result.png")
    with open(image_path, "wb") as f:
        f.write(response.content)

    return image_path