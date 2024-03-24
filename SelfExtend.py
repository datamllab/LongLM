from types import MethodType
from functools import partial
import self_extend_patch as SE

def modify_method_of_instance(instance, target_class_name, target_method_name, new_method, visited_instances=None):
    """
        This function modifies the method of an instance of a model class. 
        It's part from chat-GPT.
        It will replace the method  with the new method.
        Currently, we only use this function to modify the attention method of a model. Do not test it further. 

        instance: 
            instance of a model to modify.
        target_class_name: 
            name of the attention class to modify. E.g. 'LlamaAttention', 'GPTNeoXAttention', etc.
        new_method: new method to replace the original method. E.g. 'self_extend_forward'. 
            It should include a parameter 'self' to be binded to the instance.
    """
    target_found = False
    if visited_instances is None:
        visited_instances = set()
    # Unique identifier for the instance (using id() since object's id is unique)
    instance_id = id(instance)
    if instance_id in visited_instances:
        target_found = False
        return target_found
    # Add the instance to the already_visited set
    visited_instances.add(instance_id)

    # Check if this instance is of the target class
    if instance.__class__.__name__ == target_class_name:
        bond_method = MethodType(new_method, instance) 
        setattr(instance, target_method_name, bond_method)
        target_found = True
        return target_found
    elif hasattr(instance, '__dict__'):
        for attr_name, attr_value in instance.__dict__.items():
            if isinstance(attr_value, object) and not isinstance(attr_value, (list, tuple, dict, set)):
                _found = modify_method_of_instance(attr_value, target_class_name, target_method_name, new_method, visited_instances)
                if _found:
                    target_found = True
            elif isinstance(attr_value, (list, tuple)):
                for item in attr_value:
                    if isinstance(item, object):
                        _found = modify_method_of_instance(item, target_class_name, target_method_name, new_method, visited_instances)
                        if _found:
                            target_found = True
            # If attribute value is a dictionary, iterate over its values and recurse
            # E.g, for a ModuleList, its moudels are stored in a dictionary: ._modules
            elif isinstance(attr_value, dict):
                for key, value in attr_value.items():
                    if isinstance(value, object):
                        _found = modify_method_of_instance(value, target_class_name, target_method_name, new_method, visited_instances)
                        if _found:
                            target_found = True
            # If attribute value is a set, iterate and recurse
            elif isinstance(attr_value, set):
                for item in attr_value:
                    if isinstance(item, object):
                        _found = modify_method_of_instance(item, target_class_name, target_method_name, new_method, visited_instances)
                        if _found:
                            target_found = True

    return target_found


def apply(loaded_model, group_size, window_size, enable_flash_attention=False, scale_base=-1):
    '''
        loaded_model: 
            model to apply the self-attention extension. 
        group_size: 
            group size for the self-attention extension. 
        window_size: 
            window size for the self-attention extension. 
        scale_base:
            base for the scale, equal to pretraining length. 
            e.g. 4096 for Llama, 8192 for Gemma

            Two recommended scale factor:
                yarn: https://arxiv.org/abs/2309.00071
                log: https://arxiv.org/abs/2202.12172 ; https://kexue.fm/archives/8823
            This is helpful while retrieving a long sequence (e.g a long passkey).
            But on real-world data, the impact is minor. (e.g. on LongBench, LEval).

            The reported results in our paper does not use this scale except for long passkey retrieval.
    '''
    arch_name = loaded_model.__class__.__name__
    if 'Llama' in arch_name:
        if enable_flash_attention:
            self_extend_attention_forward = partial(SE.Llama.flash_self_extend_forward,
                                            group_size_1=group_size, 
                                            group_size_2=window_size,
                                            scale_base=scale_base)
            modifed_1 = modify_method_of_instance(loaded_model, "LlamaFlashAttention2", "_flash_attention_forward", SE.selfextend_flash_attn.flash_attention2_forward_with_window_size)
            modifed_2 = modify_method_of_instance(loaded_model, "LlamaFlashAttention2", "forward", self_extend_attention_forward)
            if (not modifed_1) or (not modifed_2):
                raise Exception(f"Failed to modify the attention method of {arch_name}")
        else:
            self_extend_attention_forward = partial(SE.Llama.self_extend_forward,
                                            group_size_1=group_size, 
                                            group_size_2=window_size,
                                            scale_base=scale_base)
            # after the default version of attention in 4.36 is LlamaSpdaAttention, but in before 4,36 or in 4.38, it is LlamaAttention
            modifed_2 = modify_method_of_instance(loaded_model, "LlamaAttention", "forward", self_extend_attention_forward)
            if not modifed_2:
                raise Exception(f"Failed to modify the attention method of {arch_name}")
    elif 'Mistral' in arch_name:
        # Mistral shares the same architecture with Llama, so the implementation should be exchangable.
        if enable_flash_attention:
            self_extend_attention_forward = partial(SE.Mistral.flash_self_extend_forward,
                                            group_size_1=group_size, 
                                            group_size_2=window_size,
                                            scale_base=scale_base)
            modifed_1 = modify_method_of_instance(loaded_model, "MistralFlashAttention2", "_flash_attention_forward", SE.selfextend_flash_attn.flash_attention2_forward_with_window_size)
            modifed_2 = modify_method_of_instance(loaded_model, "MistralFlashAttention2", "forward", self_extend_attention_forward)
            if (not modifed_1) or (not modifed_2):
                raise Exception(f"Failed to modify the attention method of {arch_name}")
        else:
            self_extend_attention_forward = partial(SE.Mistral.self_extend_forward,
                                            group_size_1=group_size, 
                                            group_size_2=window_size,
                                            scale_base=scale_base)
            modifed_2 = modify_method_of_instance(loaded_model, "MistralAttention", "forward", self_extend_attention_forward)
            if not modifed_2:
                raise Exception(f"Failed to modify the attention method of {arch_name}")
    elif 'Gemma' in arch_name:
        if enable_flash_attention:
            self_extend_attention_forward = partial(SE.Gemma.flash_self_extend_forward,
                                            group_size_1=group_size, 
                                            group_size_2=window_size,
                                            scale_base=scale_base)
            modifed_1 = modify_method_of_instance(loaded_model, "GemmaFlashAttention2", "_flash_attention_forward", SE.selfextend_flash_attn.flash_attention2_forward_with_window_size)
            modifed_2 = modify_method_of_instance(loaded_model, "GemmaFlashAttention2", "forward", self_extend_attention_forward)
            if (not modifed_1) or (not modifed_2):
                raise Exception(f"Failed to modify the attention method of {arch_name}")
        else:
            self_extend_attention_forward = partial(SE.Gemma.self_extend_forward,
                                            group_size_1=group_size,
                                            group_size_2=window_size,
                                            scale_base=scale_base)
            modifed_2= modify_method_of_instance(loaded_model, "GemmaAttention", "forward", self_extend_attention_forward)
            if not modifed_2:
                raise Exception(f"Failed to modify the attention method of {arch_name}")
    elif 'Qwen2' in arch_name:
        if enable_flash_attention:
            self_extend_attention_forward = partial(SE.Qwen2.flash_self_extend_forward,
                                            group_size_1=group_size, 
                                            group_size_2=window_size,
                                            scale_base=scale_base)
            modifed_1 = modify_method_of_instance(loaded_model, "Qwen2FlashAttention2", "_flash_attention_forward", SE.selfextend_flash_attn.flash_attention2_forward_with_window_size)
            modifed_2 = modify_method_of_instance(loaded_model, "Qwen2FlashAttention2", "forward", self_extend_attention_forward)
            if (not modifed_1) or (not modifed_2):
                raise Exception(f"Failed to modify the attention method of {arch_name}")
        else:
            self_extend_attention_forward = partial(SE.Qwen2.self_extend_forward,
                                            group_size_1=group_size, 
                                            group_size_2=window_size,
                                            scale_base=scale_base)
            modifed_2 = modify_method_of_instance(loaded_model, "Qwen2Attention", "forward", self_extend_attention_forward)
            if not modifed_2:
                raise Exception(f"Failed to modify the attention method of {arch_name}")
    elif 'Phi' in arch_name:
        if enable_flash_attention:
            self_extend_attention_forward = partial(SE.Phi.flash_self_extend_forward,
                                            group_size_1=group_size, 
                                            group_size_2=window_size,
                                            scale_base=scale_base)
            modifed_1 = modify_method_of_instance(loaded_model, "PhiFlashAttention2", "_flash_attention_forward", SE.selfextend_flash_attn.flash_attention2_forward_with_window_size)
            modifed_2 = modify_method_of_instance(loaded_model, "PhiFlashAttention2", "forward", self_extend_attention_forward)
            if (not modifed_1) or (not modifed_2):
                raise Exception(f"Failed to modify the attention method of {arch_name}")
        else:
            self_extend_attention_forward = partial(SE.Phi.self_extend_forward,
                                            group_size_1=group_size, 
                                            group_size_2=window_size,
                                            scale_base=scale_base)
            modifed_2 = modify_method_of_instance(loaded_model, "PhiAttention", "forward", self_extend_attention_forward)
            if not modifed_2:
                raise Exception(f"Failed to modify the attention method of {arch_name}")
    else:
        raise NotImplementedError

