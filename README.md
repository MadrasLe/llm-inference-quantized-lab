#  Quant-LLM-CLI: Chat Interface LLM

> **Run Large Language Models (LLMs) on accessible hardware via 4-bit Quantization.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-State%20of%20the%20Art-yellow?style=for-the-badge)
![BitsAndBytes](https://img.shields.io/badge/BitsAndBytes-Quantization-red?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

---

## About

This repository provides a **Command Line Interface (CLI)** tool designed for efficient inference and interactive chat with modern LLMs.

The main focus is **extreme efficiency**: utilizing advanced quantization techniques (NF4 via `bitsandbytes and unsloth`), this script allows you to load massive models on GPUs with limited memory (VRAM) while maintaining high performance. 

## Key Features

- ** Infinite Chat Loop:** Keeps the model loaded in memory, allowing for long sessions without reloading weights.
- ** Context Memory:** The AI remembers the current conversation history, enabling continuous interactions.
- ** 4-bit Quantization (NF4):** Reduces VRAM usage by up to 4x with few quality loss.
- ** Real-Time Streaming:** "Typewriter" effect, generating responses token-by-token instantly.
- ** Universal:** Compatible with most causal models on Hugging Face (Llama 3, Mistral, Qwen 2.5, ERNIE, Gemma, etc.).
- ** Colorful Interface:** Visual feedback in the terminal to distinguish between user, system, and error messages.

---

## Benchmarks & Tested Hardware

The point of this script is enabling you to run models that technically "shouldn't fit" on your GPU.

### 🟢 NVIDIA T4 (16GB VRAM)
*Common hardware on Google Colab Free Tier.*
Thanks to 4-bit quantization, we can allocate large models with seamless performance:
- ✅ **Baidu ERNIE-4.5-21B-Thinking** 
- ✅ **Llama-3-8B-Instruct** 
- ✅ **Mistral-7B-v0.1**

### 🟡 NVIDIA L4 (24GB VRAM)
With slightly more memory, the script scales to medium-large models:
- ✅ **Qwen-2.5-32B-Instruct**
- ✅ **Gemma-2-27B**

### 🔴 NVIDIA A100 (40GB/80GB VRAM)
For high-end infrastructure:
- ✅ **Llama-3-70B-Instruct**

---

## 🛠️ Installation

### Prerequisites
- **Python 3.10+**
- **NVIDIA GPU** with updated drivers (CUDA 11.8 support or higher).

### Step-by-Step

Install dependencies:

´´´bash
pip install -r requirements.txt
´´´


The script is flexible. You can run it with the default model (ERNIE) or specify any other Hugging Face ID.

1. Basic Usage

Loads the default model (baidu/ERNIE-4.5-21B-A3B-Thinking) and starts the chat.

´´´bash
python cli.py
´´´
2. Running a Different Model

Example running Llama 3 8B:


´´´bash
python cli.py --model "meta-llama/Meta-Llama-3-8B-Instruct"
´´´

3. Customizing Personality (System Prompt)

Define how the AI should behave from the start:

´´´bash
python cli.py --system "You are a senior Python expert who answers concisely."
´´´

4. Help and Arguments

To see all available options:


´´´bash
python cli.py --help
´´´



 License

This project is licensed under the MIT License - see the LICENSE file for details.
