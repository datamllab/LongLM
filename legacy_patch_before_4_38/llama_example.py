# transfromers version 4.32.0
import warnings
warnings.filterwarnings("ignore")

import llama_self_extend_patch as LlamaSE
from modify_utils import modify_method_of_instance
from functools import partial
import torch 
import json
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers import AutoTokenizer, AutoModelForCausalLM

original_llama_forward = LlamaAttention.forward
#self_extend_forward = partial(LlamaSE.self_extend_forward, group_size_1=8, group_size_2=1024)
self_extend_forward = partial(LlamaSE.flash_self_extend_forward,
                                                        group_size_1=8, 
                                                        group_size_2=1024)

model_path = 'meta-llama/Llama-2-7b-chat-hf'
#model_path = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir='/data/HyJ', device_map="auto", torch_dtype=torch.float16, use_flash_attention_2=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()


for line in open("passkey_examples_5k.jsonl", "r"):
    example = json.loads(line)
    prompt_postfix = "What is the pass key? The pass key is "
    prompt = example["input"] + prompt_postfix
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    print( "-----------------------------------" )
    print( f"#Tokens of Prompt:", input_ids.shape[1], end=" " )
    print( "Passkey target:", example["target"] )


    modify_method_of_instance(model, "LlamaAttention", "forward", original_llama_forward)
    tokens = model.generate(input_ids, max_new_tokens=6)
    answer= "Llama2:     [" + prompt_postfix + tokenizer.decode(tokens[0].tolist()[input_ids.shape[1]:], skip_special_tokens=True)  + "]"
    answer = answer.replace("\n", "\\n")
    print( answer )


    #modify_method_of_instance(model, "LlamaAttention", "forward", self_extend_forward)
    #modify_method_of_instance(model, "LlamaSdpaAttention", "forward", self_extend_forward)
    #modify_method_of_instance(model, "LlamaSdpaAttention", "forward", self_extend_forward)

    modify_method_of_instance(model, "LlamaFlashAttention2", "_flash_attention_forward", LlamaSE.flash_attention2_forward_with_window_size)
    modify_method_of_instance(model, "LlamaFlashAttention2", "forward", self_extend_forward)
    tokens = model.generate(input_ids, max_new_tokens=6)
    answer= "SelfExtend: [" + prompt_postfix + tokenizer.decode(tokens[0].tolist()[input_ids.shape[1]:], skip_special_tokens=True)  + "]"
    answer = answer.replace("\n", "\\n")
    print( answer )
    print( "-----------------------------------\n" )






