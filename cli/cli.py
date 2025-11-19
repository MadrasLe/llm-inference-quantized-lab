#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ERNIE 21B Quantized – Fast CLI Inference
Minimal, clean, and professional implementation.

Features:
- 4-bit quantized loading (BitsAndBytes)
- Chat template support
- Greedy decoding loop (streaming)
- Tokens-per-second benchmark
- CLI interface for any prompt / model / params

Author: Gabriel (ETL_DatasetNLP)
"""

import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ---------------------------------------------------------
# CLI PARSER
# ---------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(
        description="ERNIE 21B 4bit inference (streaming greedy generation)."
    )

    p.add_argument("--model", type=str,
                   default="baidu/ERNIE-4.5-21B-A3B-Thinking",
                   help="Model name or path")

    p.add_argument("--prompt", type=str,
                   required=True,
                   help="User input prompt")

    p.add_argument("--system", type=str,
                   default="Você é um assistente útil.",
                   help="System message for chat template")

    p.add_argument("--max-tokens", type=int,
                   default=256,
                   help="Maximum number of generated tokens")

    p.add_argument("--quant", type=str,
                   default="4bit",
                   choices=["4bit", "8bit", "none"],
                   help="Quantization type")

    return p.parse_args()


# ---------------------------------------------------------
# LOAD QUANTIZATION CONFIG
# ---------------------------------------------------------
def get_quant_config(mode):
    if mode == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif mode == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


# ---------------------------------------------------------
# MAIN INFERENCE
# ---------------------------------------------------------
def main():
    args = get_args()

    print(f"\n== ERNIE Inference ==")
    print(f"Model: {args.model}")
    print(f"Quantization: {args.quant}")
    print(f"Max tokens: {args.max_tokens}")
    print("---------------------------------\n")

    quant_config = get_quant_config(args.quant)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
    )

    # Build chat template
    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    eos = tokenizer.eos_token_id

    # Encode initial prompt
    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    prompt_len = input_ids.shape[1]

    print("Generating...\n")
    texto_decodificado_anterior = ""

    tokens_gerados = 0
    t0 = time.time()

    with torch.no_grad():
        for _ in range(args.max_tokens):

            outputs = model(input_ids=input_ids)
            next_logits = outputs.logits[:, -1, :]
            next_id = torch.argmax(next_logits, dim=-1).unsqueeze(-1)

            input_ids = torch.cat([input_ids, next_id], dim=-1)
            tokens_gerados += 1

            # Decode only newly generated text
            novos_tokens = input_ids[0][prompt_len:]
            texto = tokenizer.decode(novos_tokens, skip_special_tokens=True)

            diff = texto[len(texto_decodificado_anterior):]
            print(diff, end="", flush=True)

            texto_decodificado_anterior = texto

            if next_id.item() == eos:
                break

    t1 = time.time()
    elapsed = t1 - t0
    tps = tokens_gerados / elapsed if elapsed > 0 else 0

    print("\n\n===================================")
    print(f"Tokens gerados: {tokens_gerados}")
    print(f"Tempo total:    {elapsed:.2f} s")
    print(f"Velocidade:     {tps:.2f} tokens/s")
    print("===================================\n")


if __name__ == "__main__":
    main()
