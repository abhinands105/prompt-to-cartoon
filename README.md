# Project S8 — Generative AI Style Learning & LoRA Training

> An end-to-end local generative AI experimentation pipeline for learning visual styles with Stable Diffusion and parameter-efficient LoRA fine-tuning.

## Overview

Project S8 is a generative AI research and engineering project focused on building, training, evaluating, and deploying lightweight LoRA adapters for stylized image generation.

Instead of treating image generation as only a prompt-to-image task, the project explores the complete machine-learning workflow:

```text
Dataset Collection
        ↓
Dataset Organization
        ↓
Image Preparation
        ↓
Caption / Metadata Preparation
        ↓
Dataset Validation
        ↓
Training Configuration
        ↓
LoRA Fine-Tuning
        ↓
Checkpoint Generation
        ↓
Checkpoint Evaluation
        ↓
Local Inference
        ↓
Visual Evaluation
