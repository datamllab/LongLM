# transfromers version 4.38.2
import warnings
warnings.filterwarnings("ignore")

import torch 
import json
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

import SelfExtend 


window_size = 1024
group_size = 32
use_flash = True

model_lists = ['google/gemma-7b-it', 'meta-llama/Llama-2-7b-chat-hf', 'mistralai/Mistral-7B-Instruct-v0.1', ]

for model_name in model_lists:
    if 'Mistral' in model_name:
        # Disable Mistral's sliding window
        config = AutoConfig.from_pretrained(model_name)
        config.sliding_window = None
        model = AutoModelForCausalLM.from_pretrained(model_name, config=config, device_map="auto", torch_dtype=torch.bfloat16, use_flash_attention_2=use_flash)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, use_flash_attention_2=use_flash)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    file_name = "passkey_examples_32k.jsonl"
    #for line in open("passkey_examples_5k.jsonl", "r"):
    #for line in open("passkey_examples_10k.jsonl", "r"):
    for line in open(file_name, "r"):
        example = json.loads(line)
        prompt_postfix = "What is the pass key? The pass key is "
        prompt = example["input"] + prompt_postfix
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        print( "-----------------------------------" )
        print( f"#Tokens of Prompt:", input_ids.shape[1], end=" " )
        print( "Passkey target:", example["target"] )

        tokens = model.generate(input_ids, max_new_tokens=len(example["target"]))
        answer = prompt_postfix + tokenizer.decode(tokens[0].tolist()[input_ids.shape[1]:], skip_special_tokens=True)
        answer = answer.replace("\n", "\\n")
        answer= f"{model_name}:\n     [ {answer} ]"
        print( answer )
        print( "-----------------------------------\n" )

    
    print("=========="*2 + "**SelfExtend**" + "=========="*2)
    SelfExtend.apply(model, group_size, window_size, enable_flash_attention=use_flash)
    for line in open(file_name, "r"):
        example = json.loads(line)
        prompt_postfix = "What is the pass key? The pass key is "
        prompt = example["input"] + prompt_postfix
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        print( f"#Tokens of Prompt:", input_ids.shape[1], end=" " )
        print( "Passkey target:", example["target"] )

        tokens = model.generate(input_ids, max_new_tokens=len(example["target"]))
        answer = prompt_postfix + tokenizer.decode(tokens[0].tolist()[input_ids.shape[1]:], skip_special_tokens=True)
        answer = answer.replace("\n", "\\n")
        answer= f"SelfExtended-{model_name}:\n     [ {answer} ]"
        print( answer )
        print( "-----------------------------------\n" )
