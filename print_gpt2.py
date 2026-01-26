import pathlib
import torch
import typing as t
from transformers import AutoModelForPreTraining, AutoConfig, AutoModelForCausalLM
from selfcond.models import PytorchTransformersModel

# --- Code Snippet to Load and Print Layers ---

def load_and_print_gpt2_layers():
    """Loads GPT-2 using PytorchTransformersModel and prints all named modules."""
    
    # 1. Define model parameters
    model_name = "gpt2"
    sequence_length = 512
    cache_directory = None  # Let Hugging Face manage cache
    device = "cpu" # Use CPU to ensure compatibility across environments

    print(f"Loading model: **{model_name}**")
    
    # 2. Instantiate the model wrapper
    # This calls transformers_class_from_name internally to load the GPT-2 PyTorch model
    try:
        wrapped_model = PytorchTransformersModel(
            model_name=model_name,
            cache_dir=cache_directory,
            seq_len=sequence_length,
            device=device,
        )
    except Exception as e:
        print(f"Could not load the model: {e}")
        return

    print("-" * 60)
    print(f"**Layers in {model_name}:**")
    
    # 3. Get the underlying PyTorch module
    pytorch_module = wrapped_model.module

    # 4. Print all named modules with their parameter shapes
    total_neurons = 0
    layer_neuron_counts = {}
    
    for name, module in pytorch_module.named_modules():
        # Check for Conv1D layers (used in GPT-2 for linear projections)
        if hasattr(module, 'weight'):
            weight_shape = tuple(module.weight.shape)
            # For Conv1D in GPT-2: weight shape is (in_features, out_features)
            # The "neurons" are the output features
            if len(weight_shape) == 2:
                out_features = weight_shape[1]
                print(f"Layer: {name:50s} | Type: {type(module).__name__:15s} | Weight: {weight_shape} | Neurons: {out_features}")
                layer_neuron_counts[name] = out_features
                
    print("-" * 60)
    print("\n**Summary of sub-layers per transformer block:**")
    
    # Group by block and sub-layer type
    for block_idx in range(12):
        block_name = f"transformer.h.{block_idx}"
        print(f"\nBlock {block_idx}:")
        for name, neurons in layer_neuron_counts.items():
            if name.startswith(block_name):
                sublayer = name.replace(block_name + ".", "")
                print(f"  {sublayer}: {neurons} neurons")
                total_neurons += neurons
                
    print(f"\n**Total neurons across all blocks:** {total_neurons}")

# Execute the function
load_and_print_gpt2_layers()