# 🖌️ Prompt-to-Cartoon

**Prompt-to-Cartoon** is a fully offline generative AI system that transforms natural language prompts into cartoon or sketch-style images using diffusion and GAN-based models. Designed for artists, storytellers, and researchers, this tool provides high-quality stylized image generation without relying on external APIs.

---

## 🎯 Features

- ✏️ **Text-to-Image Generation** using locally deployed Stable Diffusion
- 🖍️ **Cartoon/Sketch Stylization** with CartoonGAN, AnimeGAN, or custom filters
- 🔒 **Fully Offline** – No APIs or cloud inference used
- 🎛️ **Interactive Interface** using Gradio
- 📦 Easy to install and run on local machines

---

## 🛠️ Tech Stack

- **Python**
- **PyTorch**
- **Stable Diffusion XL / SD 1.5 (locally fine-tuned)**
- **CartoonGAN / AnimeGAN / OpenCV stylization filters**
- **Gradio** (for UI)
- **Transformers & Tokenizers** (for optional prompt enhancement)

---

## 📸 Sample Usage

```text
Prompt: "A fox wearing glasses reading a book under a sakura tree"
→ Output: Stylized sketch/cartoon image
