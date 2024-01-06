# LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning


Offical implementation of SelfExtend of [LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning](https://arxiv.org/pdf/2401.01325.pdf). If you find our method useful, please kindly cite our paper.

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
- [01/05/2024]: Our proposed method is discussed on this [Reddit post](https://www.reddit.com/r/LocalLLaMA/s/IFOnL7yGNK) 



## 1. Overview 
This work elicits LLMs' inherent ability to handle long contexts without fine-tuning. The limited length of the training sequence during training may limit the application of Large Language Models (LLMs) on long input sequences for inference. In this work, we argue that existing LLMs themselves have inherent capabilities for handling long contexts. Based on this argument, we suggest extending LLMs' context window by themselves to fully utilize the inherent ability.We propose Self-Extend to stimulate LLMs' long context handling potential. The basic idea is to construct bi-level attention information: the group level and the neighbor level. The two levels are computed by the original model's self-attention, which means the proposed  does not require any training.

<p align="center">
<img width="600" src="./img/self_ext.jpg">


## 2. How to Use SelfExtend

### 2.1 Setup
The python packages used are:
```bash
transformers==4.32.0 
```
Note: the KV cache structure was changed after version 4.36.0 in the transformers package. You may need to modify the implementation.


### 2.2 Run
```python
import llama_self_extend_patch as LlamaSE
from modify_utils import modify_method_of_instance
from functools import partial

# Loading your model, e.g., loaded_model = AutoModelForCausalLM.from_pretrained(model_path) 

# group_size_1 is, group_size_2 is neighbor_window
self_extend_forward = partial(LlamaSE.self_extend_forward, group_size_1=4, group_size_2=1024)
modify_method_of_instance(loaded_model, "LlamaAttention", "forward", self_extend_forward)

# Inference, e.g., loaded_model.generate(...)

```


### 2.3 Passkey Example

To execute a demonstration of SelfExtend on the Passkey Retrivale task, you can use the command below:

```python
python example.py
```


Following this, you will have the following results:

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



## 4.How to choose the group_size and neighbor_window

The following thoughts are based on our experience:

- With Llama-2 as the base model, **2\~64** are reasonable for group_size; **512\~1536** are feasible for neighbor_window. But larger group_size and smaller neighbor_window are also good in many cases. 

- The general rule of choosing group_size and neighbor_window is: ensure the input sequence lenght is within the maximum extended window size (For llama-2, it would be (4096 - neighbor_window) * group_size + neighbor_window ). 

- We didn't choose the group size carefully. For the same sequence, smaller group should be better. But we found this does not strictly hold in some experiments: 
  > Sometimes, a larger group size can be beneficial. This may be due to the fact that larger positions are not well-trained. A larger group size can utilize smaller positions, which have received more training, to facilitate extension. However, smaller group sizes tend to have better precision. Thus, there is a trade-off. For more details, refer to the ablation study section. <br><br>For example:<br>If the input length for a QA task is 15,800, with a neighbor window set to 1,024, the group size can be set to 5. This is because 5 * (4,096 - 1,024) + 1,024 equals 16,384, which is greater than 15,800. However, setting the group size to 6, or even larger, such as 8 or 16, might improve the model's performance. With a group size of 5, Self-Extend uses positions 1,025 to 3,979 to extend the context window. If the group size is set to 8, Self-Extend uses positions 1,025 to 2,871 for extension. Although a group size of 8 is less precise than a group size of 5, the positions 2,872 to 3,979, utilized by a group size of 5, are less trained during pretraining, which may affect the effectiveness of the extension.

- Maybe, for a sequence of length L, you can try the smallest group size first [calculated by: G * (L- w_n) + w_n] , and then test whether larger group can be better.

## 5. Future Plan
As we stated before, our current implementation 

- [ ] Reduce redundant attention computation. 
- [ ] Useless KV-cache eviction for neighbor/normal attention.
- [ ] Flash Attention Implementation 


## 6. Contributing
We welcome contributions from the research community to improve the effeicency of SelfExtend. If you have any idea or would like to report a bug, please open an issue or submit a pull request.

## 7. License
The code is released under the MIT License.

