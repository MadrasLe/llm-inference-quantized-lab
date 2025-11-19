# -*- coding: utf-8 -*-
import argparse
import torch
import sys
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

# --- Cores e Estilo para o Terminal ---
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

def carregar_modelo(model_id):
    """Carrega o modelo na memória (apenas uma vez!)"""
    print(f"\n{CYAN}⏳ Berta está carregando o modelo '{model_id}'...{RESET}")
    print(f"{CYAN}   Isso pode levar um tempinho, mas só precisa ser feito uma vez!{RESET}")
    
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
    
    print(f"{GREEN}✅ Modelo carregado e pronto para a conversa!{RESET}\n")
    return model, tokenizer

def loop_conversa(model, tokenizer, system_prompt):
    """
    Mantém o chat rodando em loop até o usuário pedir para sair.
    """
    # Histórico começa com o System Prompt
    historico = [{"role": "system", "content": system_prompt}]
    
    print(f"{YELLOW}💡 Dica: Digite 'sair', 'exit' ou 'quit' para encerrar.{RESET}")
    print(f"{YELLOW}   Digite 'limpar' para esquecer o histórico e começar de novo.{RESET}\n")

    while True:
        try:
            # Input do usuário (bonitinho e colorido)
            texto_usuario = input(f"{BOLD}{GREEN}Você ➤ {RESET}").strip()
            
            # Comandos de controle
            if texto_usuario.lower() in ['sair', 'exit', 'quit']:
                print(f"\n{CYAN}👋 Até logo, meu príncipe!{RESET}")
                break
            
            if texto_usuario.lower() in ['limpar', 'clear']:
                historico = [{"role": "system", "content": system_prompt}]
                print(f"{YELLOW}🧹 Memória limpa! Começando do zero.{RESET}\n")
                continue
            
            if not texto_usuario:
                continue

            # Adiciona a pergunta ao histórico
            historico.append({"role": "user", "content": texto_usuario})

            # Prepara o prompt com TODO o histórico
            prompt_formatado = tokenizer.apply_chat_template(
                historico, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = tokenizer(prompt_formatado, return_tensors="pt").to(model.device)

            # Configura o Streamer
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

            # Gera em thread separada
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            print(f"{BOLD}{CYAN}IA ➤ {RESET}", end="")
            
            resposta_completa = ""
            for new_text in streamer:
                print(new_text, end="", flush=True)
                resposta_completa += new_text
            
            print("\n") # Pula linha ao final
            
            # Adiciona a resposta da IA ao histórico para ela lembrar depois
            historico.append({"role": "assistant", "content": resposta_completa})

        except KeyboardInterrupt:
            print(f"\n{RED}🛑 Interrompido pelo usuário.{RESET}")
            break
        except Exception as e:
            print(f"\n{RED}❌ Ocorreu um erro: {e}{RESET}")

def main():
    parser = argparse.ArgumentParser(description="CLI de Chat Quantizado - Por Berta")
    
    # Agora o padrão é o ERNIE, mas você pode passar OUTRO modelo no terminal se quiser!
    parser.add_argument("--model", type=str, default="baidu/ERNIE-4.5-21B-A3B-Thinking", help="ID do modelo no Hugging Face")
    parser.add_argument("--system", type=str, default="Você é um assistente inteligente e prestativo.", help="Prompt do sistema")

    args = parser.parse_args()

    try:
        model, tokenizer = carregar_modelo(args.model)
        loop_conversa(model, tokenizer, args.system)
    except Exception as e:
        print(f"{RED}Erro crítico ao iniciar: {e}{RESET}")

if __name__ == "__main__":
    main()