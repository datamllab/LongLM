# LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning


Implementation of the proposed SelfExtend in [LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning](https://arxiv.org/pdf/2401.01325.pdf). If you find our method useful, please kindly cite our paper.

```bibtex
@misc{jin2024llm,
      title={LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning}, 
      author={Hongye Jin and Xiaotian Han and Jingfeng Yang and Zhimeng Jiang and Zirui Liu and Chia-Yuan Chang and Huiyuan Chen and Xia Hu},
      year={2024},
      eprint={2401.01325},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


## Updates:
- [02/22/2024]: 🔥🔥 We added the [Implementation for Google New LLM Gemma](https://github.com/datamllab/LongLM/blob/master/gemma_self_extend_patch.py)!!! Welcome to try and test it out!!
- [01/19/2024]: We've added the [implementation for Llama with transformers 4.36.2](https://github.com/datamllab/LongLM/blob/master/llama_self_extend_patch_4_36.py) and the [implementation for microsoft's offical phi-2 with transformers 4.37](https://github.com/datamllab/LongLM/blob/master/phi_self_extend_patch_4_37.py). Another good news: the flash attention version will come in days!💥
- [01/11/2024]: We've tested the implementation for phi-2. [It works](./img/phi2_long_bench.jpg). You may find some results on this [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/194mmki/selfextend_works_for_phi2_now_looks_good/?utm_source=share&utm_medium=web2x&context=3) and details on this [X post](https://x.com/serendip410/status/1745668085711790553?s=20)
- [01/08/2024]: Add third-party implementations section
- [01/07/2024]: Add Implementation for Mistral
- [01/05/2024]: Our proposed method is discussed on this [Reddit post](https://www.reddit.com/r/LocalLLaMA/s/IFOnL7yGNK) 



## 1. Overview 
This work elicits LLMs' inherent ability to handle long contexts without fine-tuning. The limited length of the training sequence during training may limit the application of Large Language Models (LLMs) on long input sequences for inference. In this work, we argue that existing LLMs themselves have inherent capabilities for handling long contexts. Based on this argument, we suggest extending LLMs' context window by themselves to fully utilize the inherent ability. We propose Self-Extend to stimulate LLMs' long context handling potential. The basic idea is to construct bi-level attention information: the group level and the neighbor level. The two levels are computed by the original model's self-attention, which means the proposed  does not require any training.

<p align="center">
<img width="600" src="./img/self_ext.jpg">


## 2. How to Use SelfExtend

### 2.1 Setup
For current Llama Implementation, the python packages used are:
```bash
transformers==4.32.0 
```
However, the KV cache structure was changed after version 4.36.0 in the transformers package. We will updata it to 4.36 in the near future. 
You can modify the implementation by yourself if you use transformers>=4.36.0

For Mistral Implementation, the python packages used are:
```bash
transformers==4.36.2 
```
 Mistral is similar to Llama, this implementation can be a good example about how to implement Self-Extend with transformers>=4.36.0


For Gemma Implementation, the python packages used are:
```bash
transformers==4.38.1
```

### Installation

Clone the repository to your machine and copy your modeling files into the cloned repo directory.

### 2.2 Run
```python
import llama_self_extend_patch as LlamaSE
from modify_utils import modify_method_of_instance
from functools import partial

# Load your model, e.g., loaded_model = AutoModelForCausalLM.from_pretrained(model_path) 

# group_size_1 is group_window, group_size_2 is neighbor_window
self_extend_forward = partial(LlamaSE.self_extend_forward, group_size_1=4, group_size_2=1024)
modify_method_of_instance(loaded_model, "LlamaAttention", "forward", self_extend_forward)

# Inference, e.g., loaded_model.generate(...)

```

```python
import mistral_self_extend_patch as MistralSE
from modify_utils import modify_method_of_instance
from functools import partial

# Load your model, e.g., loaded_model = AutoModelForCausalLM.from_pretrained(model_path) 

# group_size_1 is group_window, group_size_2 is neighbor_window
self_extend_forward = partial(MistralSE.self_extend_forward, group_size_1=4, group_size_2=1024)
modify_method_of_instance(loaded_model, "MistralAttention", "forward", self_extend_forward)

# Inference, e.g., loaded_model.generate(...)

```


### 2.3 Passkey Example

To execute a demonstration of SelfExtend on the Passkey Retrivale task, you can use the command below:

```python
python llama_example.py # llama

python mistral_example.py # mistra
```


By running the command, you will have the following results:


For Llama
```bash
-----------------------------------
#Tokens of Prompt:  5144 Passkey target:  89427
Llama2:     [What is the pass key? The pass key is .\n.\n.\n.]
SelfExtend: [What is the pass key? The pass key is 89427.]
-----------------------------------

-----------------------------------
#Tokens of Prompt:  5144 Passkey target:  51906
Llama2:     [What is the pass key? The pass key is .\n.\n.\n.]
SelfExtend: [What is the pass key? The pass key is 51906.]
-----------------------------------

-----------------------------------
#Tokens of Prompt:  5144 Passkey target:  38117
Llama2:     [What is the pass key? The pass key is \n.\n.\n.\n]
SelfExtend: [What is the pass key? The pass key is 38117.]
-----------------------------------

-----------------------------------
#Tokens of Prompt:  5144 Passkey target:  60151
Llama2:     [What is the pass key? The pass key is .\n.\n.\n.]
SelfExtend: [What is the pass key? The pass key is 60151.]
-----------------------------------

-----------------------------------
#Tokens of Prompt:  5144 Passkey target:  23789
Llama2:     [What is the pass key? The pass key is .\n.\n.\n.]
SelfExtend: [What is the pass key? The pass key is 23789.]
-----------------------------------
```

For Mistral
```bash
-----------------------------------
#Tokens of Prompt: 9994 Passkey target: 51013
Mistral:    [What is the pass key? The pass key is \n\n\n\n\n\n]
SelfExtend: [What is the pass key? The pass key is 51013.]
-----------------------------------

-----------------------------------
#Tokens of Prompt: 9994 Passkey target: 36920
Mistral:    [What is the pass key? The pass key is \n\n\n\n\n\n]
SelfExtend: [What is the pass key? The pass key is 36920.]
-----------------------------------

-----------------------------------
#Tokens of Prompt: 9994 Passkey target: 83493
Mistral:    [What is the pass key? The pass key is \n\n\n\n\n\n]
SelfExtend: [What is the pass key? The pass key is 83493.]
-----------------------------------

-----------------------------------
#Tokens of Prompt: 9994 Passkey target: 78585
Mistral:    [What is the pass key? The pass key is \n\n\n\n\n\n]
SelfExtend: [What is the pass key? The pass key is 78585.]
-----------------------------------

-----------------------------------
#Tokens of Prompt: 9994 Passkey target: 58328
Mistral:    [What is the pass key? The pass key is \n\n\n\n\n\n]
SelfExtend: [What is the pass key? The pass key is 58328.]
-----------------------------------
```



## 4.How to choose the group_size and neighbor_window

The following thoughts are based on our experience:

- With Llama-2 as the base model, **2\~64** are reasonable for group_size; **512\~1536** are feasible for neighbor_window. But larger group_size and smaller neighbor_window are also good in many cases. 

- The general rule of choosing group_size and neighbor_window is: ensure the input sequence lenght is within the maximum extended window size (For llama-2, it would be (4096 - neighbor_window) * group_size + neighbor_window ). 

- We didn't choose the group size carefully. For the same sequence, smaller group should be better. But we found this does not strictly hold in some experiments: 
  > Sometimes, a larger group size can be beneficial. This may be due to the fact that larger positions are not well-trained. A larger group size can utilize smaller positions, which have received more training, to facilitate extension. However, smaller group sizes tend to have better precision. Thus, there is a trade-off. For more details, refer to the ablation study section. <br><br>For example:<br>If the input length for a QA task is 15,800, with a neighbor window set to 1,024, the group size can be set to 5. This is because 5 * (4,096 - 1,024) + 1,024 equals 16,384, which is greater than 15,800. However, setting the group size to 6, or even larger, such as 8 or 16, might improve the model's performance. With a group size of 5, Self-Extend uses positions 1,025 to 3,979 to extend the context window. If the group size is set to 8, Self-Extend uses positions 1,025 to 2,871 for extension. Although a group size of 8 is less precise than a group size of 5, the positions 2,872 to 3,979, utilized by a group size of 5, are less trained during pretraining, which may affect the effectiveness of the extension.

- Maybe, for a sequence of length L, you can try the smallest group size first [calculated by: G * (L- w_n) + w_n] , and then test whether larger group can be better.

## 5. Future Plan
Our current implementation is primarily focused on helping readers easily understand the proposed method, and it aligns with the pseudocode. Its efficiency is not yet optimal. In the future, we plan to:

- [ ] Reduce redundant attention computation. 
- [ ] Useless KV-cache eviction for neighbor/normal attention.
- [ ] Flash Attention Implementation 


## 6. Third-party Implementations
- [https://github.com/sdan/selfextend](https://github.com/sdan/selfextend)
- [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/blob/1fc2f265ff9377a37fd2c61eae9cd813a3491bea/examples/main/main.cpp#L552)

Note: We do not test these third-party implementations, but you can try them out!

## 7. Contributing
We welcome contributions from the research community to improve the effeicency of SelfExtend. If you have any idea or would like to report a bug, please open an issue or submit a pull request.

## 8. License
The code is released under the MIT License.

