import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===============================
# DEVICE
# ===============================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ===============================
# MODEL PATH
# ===============================

MAGICPROMPT_PATH = r"C:\Users\abhis\OneDrive\Desktop\project s8\models\magicprompt"

print("Loading MagicPrompt model...")

# ===============================
# TOKENIZER
# ===============================

tokenizer = AutoTokenizer.from_pretrained(
    MAGICPROMPT_PATH,
    local_files_only=True
)

tokenizer.pad_token = tokenizer.eos_token

# ===============================
# MODEL
# ===============================

model = AutoModelForCausalLM.from_pretrained(
    MAGICPROMPT_PATH,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    local_files_only=True
).to(device)

model.eval()

print("MagicPrompt loaded successfully")

# ===============================
# PROMPT CLEANER
# ===============================

def clean_prompt(text):

    banned_words = [
        "artstation",
        "concept art",
        "oil painting",
        "greg rutkowski",
        "magali villeneuve",
        "artgerm",
        "stanley lau",
        "4k",
        "8k",
        "high definition",
        "insanely detailed",
        "hyper detailed",
        "photorealistic",
        "realistic"
    ]

    text = text.lower()

    for word in banned_words:
        text = text.replace(word, "")

    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s+", ", ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r",+", ",", text)

    return text.strip(" ,")

# ===============================
# PROMPT ENHANCER
# ===============================

def enhance_prompt(user_prompt):

    inputs = tokenizer(
        user_prompt,
        return_tensors="pt",
        padding=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():

        output = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=70,
            do_sample=True,
            temperature=0.8,
            top_k=40,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )

    enhanced = tokenizer.decode(output[0], skip_special_tokens=True)

    enhanced = clean_prompt(enhanced)

    return enhanced

# ===============================
# FINAL PROMPT BUILDER
# ===============================

def build_final_prompt(user_prompt):

    enhanced = enhance_prompt(user_prompt)

    final_prompt = (
        "masterpiece, best quality, cartoon illustration, "
        "(advtime_style:1.35), "
        "Adventure Time animation style, cel shading, flat colors, bold outlines, "
        + enhanced
    )

    return enhanced, final_prompt