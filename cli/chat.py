#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
import sys
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

# --- Terminal Colors and Style ---
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

def load_model(model_id):
    """Loads the model into memory (runs only once)."""
    print(f"\n{CYAN}⏳ Loading model '{model_id}'...{RESET}")
    print(f"{CYAN}   This may take a moment.{RESET}")
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    print(f"{GREEN}✅ Model loaded and ready!{RESET}\n")
    return model, tokenizer

def chat_loop(model, tokenizer, system_prompt):
    """
    Main chat loop. Handles input, history, and streaming response.
    """
    # Initialize history with System Prompt
    history = [{"role": "system", "content": system_prompt}]
    
    print(f"{YELLOW}💡 Hint: Type 'exit' or 'quit' to end the session.{RESET}")
    print(f"{YELLOW}   Type 'clear' to reset conversation history.{RESET}\n")

    while True:
        try:
            # User Input
            user_input = input(f"{BOLD}{GREEN}User ➤ {RESET}").strip()
            
            # Control Commands
            if user_input.lower() in ['exit', 'quit']:
                print(f"\n{CYAN}👋 Exiting.{RESET}")
                break
            
            if user_input.lower() in ['clear', 'reset']:
                history = [{"role": "system", "content": system_prompt}]
                print(f"{YELLOW}🧹 History cleared. Starting fresh.{RESET}\n")
                continue
            
            if not user_input:
                continue

            # Append user message to history
            history.append({"role": "user", "content": user_input})

            # Prepare prompt with full history
            formatted_prompt = tokenizer.apply_chat_template(
                history, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

            # Configure Streamer
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            generation_kwargs = dict(
                inputs, 
                streamer=streamer, 
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id
            )

            # Generate in a separate thread to allow streaming
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            print(f"{BOLD}{CYAN}Assistant ➤ {RESET}", end="")
            
            full_response = ""
            for new_text in streamer:
                print(new_text, end="", flush=True)
                full_response += new_text
            
            print("\n") # New line at the end
            
            # Append assistant response to history
            history.append({"role": "assistant", "content": full_response})

        except KeyboardInterrupt:
            print(f"\n{RED}🛑 Interrupted by user.{RESET}")
            break
        except Exception as e:
            print(f"\n{RED}❌ An error occurred: {e}{RESET}")

def main():
    parser = argparse.ArgumentParser(description="Interactive Quantized Chat CLI")
    
    parser.add_argument("--model", type=str, default="baidu/ERNIE-4.5-21B-A3B-Thinking", 
                        help="Hugging Face Model ID")
    parser.add_argument("--system", type=str, default="You are a helpful and intelligent assistant.", 
                        help="System Prompt")

    args = parser.parse_args()

    try:
        model, tokenizer = load_model(args.model)
        chat_loop(model, tokenizer, args.system)
    except Exception as e:
        print(f"{RED}Critical error during startup: {e}{RESET}")

if __name__ == "__main__":
    main()
