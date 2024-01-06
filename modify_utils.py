from types import MethodType


def modify_method_of_instance(instance, target_class_name, target_method_name, new_method, visited_instances=None):
    """
        This function modifies the attention method of an instance of a model. 
        It will replace the forward method of the attention class with the new method.

        instance: instance of a model to modify.
        target_class_name: name of the attention class to modify. E.g. 'LlamaAttention', 'GPTNeoXAttention', etc.
        new_method: new method to replace the original attention method. E.g. 'forward_with_rerope'. It should include a parameter 'self' to be binded to the instance.
    """
    # super important, for Circular References always happens in Python.
    # Common Data Structures: Python's data structures like lists and dictionaries can contain references to themselves.
    # Common Patterns in Libraries/Frameworks: Frameworks like PyTorch and TensorFlow can have objects that reference each other in a complex manner.
    # This might not be evident to the user, but when recursively inspecting the attributes of such objects, one can easily stumble upon circular references.
    if visited_instances is None:
        visited_instances = set()
    # Unique identifier for the instance (using id() since object's id is unique)
    instance_id = id(instance)
    if instance_id in visited_instances:
        return 
    # Add the instance to the already_visited set
    visited_instances.add(instance_id)

    # Check if this instance is of the target class
    if instance.__class__.__name__ == target_class_name:
        bond_method = MethodType(new_method, instance) 
        # original_method = getattr(instance, target_method_name)
        setattr(instance, target_method_name, bond_method)
        # original_method = getattr(instance, target_method_name)
        # return original_method
    elif hasattr(instance, '__dict__'):
        for attr_name, attr_value in instance.__dict__.items():
            if isinstance(attr_value, object) and not isinstance(attr_value, (list, tuple, dict, set)):
                modify_method_of_instance(attr_value, target_class_name, target_method_name, new_method, visited_instances)
            elif isinstance(attr_value, (list, tuple)):
                for item in attr_value:
                    if isinstance(item, object):
                        modify_method_of_instance(item, target_class_name, target_method_name, new_method, visited_instances)
            # If attribute value is a dictionary, iterate over its values and recurse
            # E.g, for a ModuleList, its moudels are stored in a dictionary: ._modules
            elif isinstance(attr_value, dict):
                for key, value in attr_value.items():
                    if isinstance(value, object):
                        modify_method_of_instance(value, target_class_name, target_method_name, new_method, visited_instances)
            
            # If attribute value is a set, iterate and recurse
            elif isinstance(attr_value, set):
                for item in attr_value:
                    if isinstance(item, object):
                        modify_method_of_instance(item, target_class_name, target_method_name, new_method, visited_instances)
    
    # return getattr(instance, target_method_name)