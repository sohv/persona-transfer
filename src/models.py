"""
Model loading and inference for cross-model persona transfer.
Supports HuggingFace transformers with direct activation injection.

Supported models:
- Qwen 2.5 7B Instruct
- LLaMA 3.1 8B Instruct
- Mistral 7B Instruct v0.3
"""
import logging
import time
import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

# Configure logging
logger = logging.getLogger(__name__)

# Available models
AVAILABLE_MODELS = {
    "qwen2.5-7b-instruct": {
        "model_type": "causal_lm",
        "model_class": "AutoModelForCausalLM",
        "tokenizer_class": "AutoTokenizer",
        "path": "Qwen/Qwen2.5-7B-Instruct",
        "description": "Qwen2.5-7B-Instruct (7B Parameters)",
        "quantization": "none",
        "max_length": 4096,
        "device_map": "auto"
    },
    "llama-3.1-8b-instruct": {
        "model_type": "causal_lm",
        "model_class": "AutoModelForCausalLM",
        "tokenizer_class": "AutoTokenizer",
        "path": "meta-llama/Llama-3.1-8B-Instruct",
        "description": "Llama-3.1-8B-Instruct (8B Parameters)",
        "quantization": "none",
        "max_length": 4096,
        "device_map": "auto"
    },
    "mistral-7b-instruct-v0.3": {
        "model_type": "causal_lm",
        "model_class": "AutoModelForCausalLM",
        "tokenizer_class": "AutoTokenizer",
        "path": "mistralai/Mistral-7B-Instruct-v0.3",
        "description": "Mistral-7B-Instruct-v0.3 (7B Parameters)",
        "quantization": "none",
        "max_length": 4096,
        "device_map": "auto"
    }
}

# Store loaded models and tokenizers
_loaded_models = {}
_loaded_tokenizers = {}
_model_configs = {}

def _get_quantization_config():
    """Get BitsAndBytesConfig for 4-bit quantization."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )

def _get_model_config(model_id):
    """Get model configuration for a given model ID."""
    if model_id not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_id} not available. Available: {list(AVAILABLE_MODELS.keys())}")
    
    return AVAILABLE_MODELS[model_id]

async def get_available_models():
    """Get list of available models."""
    return [
        {
            "id": model_id,
            "name": config["description"],
            "type": config["model_type"]
        }
        for model_id, config in AVAILABLE_MODELS.items()
        if config["model_type"] != "info"  # Filter out info-only entries
    ]

def _load_model(model_id):
    """
    Load a model and tokenizer supporting both HuggingFace and GGUF formats.
    """
    if model_id in _loaded_models:
        logger.info(f"Model {model_id} already loaded, returning cached version")
        return _loaded_models[model_id], _loaded_tokenizers[model_id]

    # MEMORY MANAGEMENT: Unload other models before loading large models
    # This prevents GPU out of memory errors
    if len(_loaded_models) > 0:
        logger.info(f"Unloading {len(_loaded_models)} existing model(s) to free memory...")
        models_to_unload = list(_loaded_models.keys())
        for existing_model_id in models_to_unload:
            logger.info(f"  Unloading {existing_model_id}...")
            unload_model(existing_model_id)

        # Clear GPU cache
        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA memory cache")

    config = _get_model_config(model_id)
    logger.info(f"Loading {model_id}...")
    return _load_hf_model(model_id, config)

def _load_hf_model(model_id, config):
    """Load HuggingFace transformers model."""
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config["path"],
        trust_remote_code=True,
        padding_side="left"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Loading model (this may take a few minutes for first download)...")
    
    # Determine best device
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16
        logger.info("Using CUDA GPU")
    else:
        device = "cpu"
        torch_dtype = torch.float32
        logger.info("Using CPU (GPU acceleration not available)")
    
    try:
        # Load model with appropriate device placement
        model = AutoModelForCausalLM.from_pretrained(
            config["path"],
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto" if device == "cuda" else None
        )
        if device != "cuda":
            model = model.to(device)
        logger.info(f"Successfully loaded {config['path']} on {device}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e
    
    # Enable gradient checkpointing to save memory
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    _loaded_models[model_id] = model
    _loaded_tokenizers[model_id] = tokenizer
    
    logger.info(f"Successfully loaded {model_id}")
    return model, tokenizer

def _attach_activation_hooks(model, model_id):
    """
    Attach hooks to extract activations from transformer layers.
    
    For GPT-OSS (causal LM), we hook the transformer blocks.
    Architecture: model.transformer.h[i] contains the transformer layers.
    """
    activations = {}
    config = _get_model_config(model_id)
    
    def hook_fn(layer_name):
        def hook(module, input, output):
            # For causal LM, output is typically a tuple (hidden_states, ...)
            # or just hidden_states tensor
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Extract activations from the last token position (sequence dimension)
            # Shape: [batch_size, sequence_length, hidden_size]
            if len(hidden_states.shape) == 3:
                # Take the last token's hidden states
                last_token_activations = hidden_states[:, -1, :]  # [batch_size, hidden_size]
                activations[layer_name] = last_token_activations.detach().clone().cpu().numpy()
            else:
                # Fallback: take the whole tensor
                activations[layer_name] = hidden_states.detach().clone().cpu().numpy()
                
        return hook
    
    hooks = []
    
    # Inspect the model structure first
    logger.info("Inspecting model architecture...")
    logger.info(f"Model type: {type(model)}")
    logger.info(f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')][:10]}")
    
    # Try multiple possible layer access patterns for different architectures
    transformer_layers = None
    layer_path = None
    
    # Pattern 1: model.transformer.h (GPT-style)
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        transformer_layers = model.transformer.h
        layer_path = "model.transformer.h"
    # Pattern 2: model.model.layers (Llama-style)
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        transformer_layers = model.model.layers
        layer_path = "model.model.layers"
    # Pattern 3: model.h (Direct access)
    elif hasattr(model, 'h'):
        transformer_layers = model.h
        layer_path = "model.h"
    # Pattern 4: model.layers (Direct layers)
    elif hasattr(model, 'layers'):
        transformer_layers = model.layers
        layer_path = "model.layers"
    
    if transformer_layers is not None:
        logger.info(f"Found {len(transformer_layers)} transformer layers at {layer_path}")
        
        for i, layer in enumerate(transformer_layers):
            hook = layer.register_forward_hook(hook_fn(f"layer_{i}"))
            hooks.append(hook)
            
        logger.info(f"Successfully hooked {len(hooks)} transformer layers")
    else:
        # Manual discovery as fallback
        logger.warning("Standard transformer layer paths not found, attempting manual discovery...")
        layer_count = 0
        
        for name, module in model.named_modules():
            # Look for transformer blocks/layers
            if any(pattern in name.lower() for pattern in ['layer', 'block', 'h.']):
                # Skip embedding, norm, and other non-transformer layers
                if not any(skip in name.lower() for skip in ['embed', 'norm', 'ln', 'head', 'pooler']):
                    logger.info(f"Discovered potential layer: {name}")
                    hook = module.register_forward_hook(hook_fn(f"discovered_{name}"))
                    hooks.append(hook)
                    layer_count += 1
                    
                    # Limit to reasonable number of layers
                    if layer_count >= 50:
                        break
        
        logger.info(f"Discovered and hooked {layer_count} potential layers")
    
    return activations, hooks

def _detach_hooks(hooks):
    """Remove hooks from the model."""
    for hook in hooks:
        hook.remove()

def _generate_text_response(model, tokenizer, model_id, system_prompt, user_prompt, max_new_tokens=150, persona_vectors=None, steering_coefficient=0.0, source_model_id=None):
    """
    Generate text response using HuggingFace models with optional persona vector steering.

    Args:
        model: The loaded model
        tokenizer: The tokenizer for the model
        model_id: Model identifier (also target model ID)
        system_prompt: System prompt to guide behavior
        user_prompt: User's question/prompt
        max_new_tokens: Maximum number of new tokens to generate
        persona_vectors: Dict of layer_name -> numpy vector for steering
        steering_coefficient: Strength of steering (-2.0 to 2.0)
        source_model_id: Optional source model ID for cross-model transfer (dimension mapping)

    Returns:
        Generated response text
    """
    # Detect if this is trait generation (needs lower temperature for JSON)
    is_trait_generation = "Generate the complete trait artifacts" in user_prompt

    return _generate_hf_response(model, tokenizer, model_id, system_prompt, user_prompt, max_new_tokens, persona_vectors, steering_coefficient, is_trait_generation, source_model_id)

def _generate_hf_response(model, tokenizer, model_id, system_prompt, user_prompt, max_new_tokens=150, persona_vectors=None, steering_coefficient=0.0, is_trait_generation=False, source_model_id=None):
    """Generate response using HuggingFace transformers model."""
    # Construct the full prompt based on model type
    if "qwen" in model_id.lower():
        # Qwen chat format
        full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    elif "llama" in model_id.lower():
        # Llama chat format
        full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "mistral" in model_id.lower():
        # Mistral chat format
        full_prompt = f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"
    else:
        raise ValueError(f"Unsupported model: {model_id}. Only qwen, llama, and mistral models are supported.")
    
    # Tokenize the input
    inputs = tokenizer(
        full_prompt, 
        return_tensors="pt", 
        truncation=True,
        max_length=1024,  # Leave room for generation
        padding=False
    )
    
    # Move inputs to the same device as the model
    if hasattr(model, 'device'):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Setup steering: Chen et al. activation injection for Qwen when steering, neutral otherwise
    steering_hooks = []
    
    if steering_coefficient == 0.0 or not persona_vectors:
        # NO STEERING: Completely neutral generation (baseline)
        logger.info(f"Using NO STEERING (neutral baseline) for {model_id}")
        # Temperature varies by use case
        if is_trait_generation:
            temperature = 0.5  # Lower for JSON consistency in trait generation
        else:
            temperature = 0.7  # Higher for personality expression in steering tests
        top_p = 0.9
        repetition_penalty = 1.1

    elif any(model_name in model_id.lower() for model_name in ["qwen", "llama", "mistral"]):
        # Use Chen et al. layer 20 activation injection for Qwen/Llama/Mistral models when steering
        # These models share the same model.model.layers architecture and support direct activation hooks
        logger.info(f"Using Chen et al. activation injection for {model_id}")
        steering_hooks = _setup_chen_layer20_injection(
            model,
            persona_vectors,
            steering_coefficient,
            source_model_id=source_model_id,
            target_model_id=model_id
        )
        # Temperature varies by use case
        if is_trait_generation:
            temperature = 0.5  # Lower for JSON consistency in trait generation
        else:
            temperature = 0.7  # Higher for personality expression with steering
        top_p = 0.9
        repetition_penalty = 1.1
        
    else:
        # Use parameter-based steering for non-Qwen models (fallback)
        logger.info(f"Using parameter-based steering for {model_id}")
        base_temp = 0.3
        base_top_p = 0.8         
        base_rep_penalty = 1.05
        
        target_layer = len(list(persona_vectors.keys())) // 2
        target_layer_name = f"layer_{target_layer}"
        
        if target_layer_name in persona_vectors:
            vector = persona_vectors[target_layer_name]
            if isinstance(vector, np.ndarray):
                magnitude_factor = min(np.linalg.norm(vector) / 100.0, 1.5)
                
                if steering_coefficient > 0:
                    temperature = min(base_temp * (1 + magnitude_factor * abs(steering_coefficient) * 1.5), 0.9)
                    top_p = min(base_top_p * (1 + abs(steering_coefficient) * 0.2), 0.95)
                    repetition_penalty = max(base_rep_penalty * (1 - abs(steering_coefficient) * 0.1), 0.9)
                else:
                    temperature = max(base_temp * (1 - magnitude_factor * abs(steering_coefficient) * 0.5), 0.1)
                    top_p = max(base_top_p * (1 - abs(steering_coefficient) * 0.2), 0.5)
                    repetition_penalty = min(base_rep_penalty * (1 + abs(steering_coefficient) * 0.2), 1.3)
                
                logger.info(f"HF parameter steering: temp={temperature:.2f}, top_p={top_p:.2f}, rep_penalty={repetition_penalty:.2f}")
            else:
                temperature, top_p, repetition_penalty = base_temp, base_top_p, base_rep_penalty
        else:
            temperature, top_p, repetition_penalty = base_temp, base_top_p, base_rep_penalty
    
    try:
        from transformers import GenerationConfig
        
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=50,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        # Decode only the new tokens (exclude the input prompt)
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Clean up common generation artifacts
        response = response.replace("Human:", "").replace("Assistant:", "").strip()
        
        return response
        
    except Exception as e:
        logger.error(f"HF generation error: {e}")
        raise e
        
    finally:
        # Always remove Chen et al. steering hooks after generation
        if steering_hooks:
            for hook in steering_hooks:
                hook.remove()
            logger.info(f"Removed {len(steering_hooks)} Chen et al. steering hooks")

def _setup_minimal_steering_hook(model, target_layer_name, persona_vector, steering_coefficient):
    """
    Setup a single, minimal steering hook following research paper approach.
    Only applies to final token position to avoid attention interference.
    """
    hooks = []
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    
    # Convert persona vector to tensor with correct dtype and device
    if isinstance(persona_vector, np.ndarray):
        steering_vector = torch.from_numpy(persona_vector).to(dtype=model_dtype, device=device)
        steering_vector = steering_vector * steering_coefficient
        logger.info(f"  Prepared minimal steering vector: shape {steering_vector.shape}, dtype {steering_vector.dtype}")
    
    def minimal_steering_hook(module, input, output):
        """
        Minimal hook that only modifies the LAST token's hidden states.
        Following research paper: h_ℓ ← h_ℓ + α·v_ℓ at final position.
        """
        try:
            # Handle different output formats
            if isinstance(output, tuple):
                hidden_states = output[0]
                other_outputs = output[1:]
            else:
                hidden_states = output
                other_outputs = ()
            
            if len(hidden_states.shape) != 3:
                return output  # Skip if unexpected shape
                
            batch_size, seq_len, hidden_size = hidden_states.shape
            
            # Only modify the LAST token position (like research paper)
            if steering_vector.shape[0] == hidden_size:
                # Clone to avoid modifying original
                modified_hidden_states = hidden_states.clone()
                # Apply steering only to final token: h[-1] += α·v
                modified_hidden_states[:, -1, :] += steering_vector
                
                logger.debug(f"Applied minimal steering to final token position")
                
                if other_outputs:
                    return (modified_hidden_states,) + other_outputs
                else:
                    return modified_hidden_states
            
            return output
                
        except Exception as e:
            logger.error(f"Error in minimal steering hook: {e}")
            return output
    
    # Apply hook to target layer only
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        transformer_layers = model.transformer.h
        layer_idx = int(target_layer_name.split('_')[1])
        
        if layer_idx < len(transformer_layers):
            hook_handle = transformer_layers[layer_idx].register_forward_hook(minimal_steering_hook)
            hooks.append(hook_handle)
            logger.info(f"  Applied minimal steering hook to {target_layer_name}")
    
    return hooks

def _setup_steering_hooks(model, persona_vectors, steering_coefficient, source_model_id=None, target_model_id=None):
    """
    Setup forward hooks to inject persona vectors during generation.

    Automatically handles dimension mismatch by applying learned mappings
    when transferring vectors across models with different hidden sizes.

    Args:
        model: The GPT-2 model
        persona_vectors: Dict of layer_name -> numpy array
        steering_coefficient: Strength multiplier for vectors
        source_model_id: Optional source model ID for dimension mapping
        target_model_id: Optional target model ID for dimension mapping

    Returns:
        List of hook handles to remove later
    """
    hooks = []
    device = next(model.parameters()).device

    logger.info(f"Setting up steering hooks for {len(persona_vectors)} layers...")

    # Detect dimension mismatch and apply mapping if needed
    if persona_vectors:
        first_vector = next(iter(persona_vectors.values()))
        source_dim = first_vector.shape[0] if isinstance(first_vector, np.ndarray) else len(first_vector)

        # Get model's hidden size
        if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
            target_dim = model.config.hidden_size
        elif hasattr(model, 'config') and hasattr(model.config, 'n_embd'):
            target_dim = model.config.n_embd
        else:
            # Fallback: detect from model parameters
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                # Try to get dimension from first layer
                first_layer = model.transformer.h[0]
                if hasattr(first_layer, 'ln_1') and hasattr(first_layer.ln_1, 'weight'):
                    target_dim = first_layer.ln_1.weight.shape[0]
                else:
                    target_dim = source_dim  # Assume match if can't detect
            else:
                target_dim = source_dim

        logger.info(f"Detected dimensions: source={source_dim}, target={target_dim}")

        # Apply dimension mapping if needed
        if source_dim != target_dim:
            logger.warning(f"Dimension mismatch detected: {source_dim} → {target_dim}")

            if source_model_id and target_model_id:
                try:
                    from dimension_mapper import get_dimension_mapper
                    mapper = get_dimension_mapper()

                    logger.info(f"Applying dimension mapping: {source_model_id}({source_dim}) → {target_model_id}({target_dim})")
                    persona_vectors = mapper.apply_mapping(
                        persona_vectors,
                        source_model_id,
                        target_model_id,
                        source_dim,
                        target_dim
                    )
                    logger.info("Successfully applied dimension mapping")
                except Exception as e:
                    logger.error(f"Failed to apply dimension mapping: {e}")
                    logger.error(f"Steering will be skipped due to dimension mismatch")
                    return []  # Return empty hooks list
            else:
                logger.error(f"Cannot apply dimension mapping: source_model_id or target_model_id not provided")
                logger.error(f"Steering will be skipped. Provide model IDs to enable cross-model transfer.")
                return []  # Return empty hooks list

    # Convert persona vectors to tensors on the right device with correct dtype
    vector_tensors = {}
    model_dtype = next(model.parameters()).dtype  # Get model's dtype (float16 or float32)
    logger.info(f"Model dtype: {model_dtype}, Device: {device}")
    
    for layer_name, vector in persona_vectors.items():
        if isinstance(vector, np.ndarray):
            # Convert to tensor with model's dtype and move to model device
            vector_tensor = torch.from_numpy(vector).to(dtype=model_dtype, device=device)
            vector_tensors[layer_name] = vector_tensor * steering_coefficient
            logger.info(f"  Prepared steering vector for {layer_name}: shape {vector_tensor.shape}, dtype {vector_tensor.dtype}")
    
    def create_steering_hook(layer_name, steering_vector):
        def steering_hook(module, input, output):
            """
            Hook that adds persona vector to layer output during forward pass.
            """
            try:
                # Handle different output formats
                if isinstance(output, tuple):
                    hidden_states = output[0]  # First element is usually hidden states
                    other_outputs = output[1:]  # Keep other outputs unchanged
                else:
                    hidden_states = output
                    other_outputs = ()
                
                # Debug: Log tensor shapes
                # Reduced logging to avoid spam
                if layer_name == 'layer_0':  # Only log first layer
                    logger.info(f"  Hook {layer_name}: hidden_states shape = {hidden_states.shape}, steering_vector shape = {steering_vector.shape}")
                
                # Get the shape of hidden states: [batch_size, seq_len, hidden_size]
                if len(hidden_states.shape) != 3:
                    logger.warning(f"Unexpected hidden_states shape in {layer_name}: {hidden_states.shape} (expected 3D)")
                    return output
                    
                batch_size, seq_len, hidden_size = hidden_states.shape
                
                # Ensure steering vector matches the hidden dimension
                if len(steering_vector.shape) != 1 or steering_vector.shape[0] != hidden_size:
                    logger.warning(f"Vector shape mismatch in {layer_name}: vector={steering_vector.shape}, expected=[{hidden_size}]")
                    return output  # Return unchanged if dimensions don't match
                
                # Simple broadcasting approach: add steering vector to each position
                # steering_vector: [hidden_size] will broadcast to [batch_size, seq_len, hidden_size]
                # Apply steering with minimal tensor modification
                # Only modify the hidden states, don't change tensor properties
                steering_addition = steering_vector.view(1, 1, -1)  # [hidden_size] -> [1, 1, hidden_size]
                steered_hidden_states = hidden_states + steering_addition  # Simple addition
                
                # Return in the same format as the input
                if other_outputs:
                    return (steered_hidden_states,) + other_outputs
                else:
                    return steered_hidden_states
                    
            except Exception as e:
                logger.error(f"Error in steering hook for {layer_name}: {e}")
                logger.error(f"  hidden_states shape: {getattr(hidden_states, 'shape', 'None')}")
                logger.error(f"  steering_vector shape: {getattr(steering_vector, 'shape', 'None')}")
                import traceback
                logger.error(f"  Full traceback: {traceback.format_exc()}")
                return output  # Return unchanged on error
        
        return steering_hook
    
    # Apply hooks to transformer layers (only middle layers to avoid attention interference)
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        transformer_layers = model.transformer.h
        total_layers = len(transformer_layers)
        
        # Only apply steering to middle layers (most effective for personality)
        # Use fewer layers to reduce chance of breaking attention mechanism
        if total_layers >= 20:
            effective_layers = [8, 12, 16]  # Middle layers for large models
        elif total_layers >= 12:
            effective_layers = [4, 6, 8]    # Middle layers for medium models  
        else:
            effective_layers = [2, 4]       # Middle layers for small models
        
        logger.info(f"Applying steering to layers: {effective_layers} out of {total_layers} total layers")
        
        for i, layer in enumerate(transformer_layers):
            layer_name = f"layer_{i}"
            
            # Only apply steering to selected effective layers
            if layer_name in vector_tensors and i in effective_layers:
                steering_vector = vector_tensors[layer_name]
                hook_fn = create_steering_hook(layer_name, steering_vector)
                hook_handle = layer.register_forward_hook(hook_fn)
                hooks.append(hook_handle)
                logger.info(f"  Applied steering hook to {layer_name} (effective layer)")
            elif layer_name in vector_tensors:
                logger.debug(f"  Skipped steering hook for {layer_name} (not in effective layers)")
    
    else:
        logger.warning("Could not find transformer layers for steering hook attachment")
    
    return hooks

def _setup_chen_layer20_injection(model, persona_vectors, steering_coefficient, source_model_id=None, target_model_id=None):
    """
    Chen et al. (2024) style layer 20 activation injection for HuggingFace models.
    Injects persona vector directly into layer 20 activations during forward pass.

    Automatically handles dimension mismatch by applying learned mappings
    when transferring vectors across models with different hidden sizes.

    Args:
        model: HuggingFace transformer model (e.g., Qwen2.5-7B)
        persona_vectors: Dict of layer_name -> numpy array
        steering_coefficient: Strength multiplier for vector injection
        source_model_id: Optional source model ID for dimension mapping
        target_model_id: Optional target model ID for dimension mapping

    Returns:
        List of hook handles to remove later
    """
    hooks = []

    # Detect dimension mismatch and apply mapping if needed
    if persona_vectors:
        first_vector = next(iter(persona_vectors.values()))
        source_dim = first_vector.shape[0] if isinstance(first_vector, np.ndarray) else len(first_vector)

        # Get model's hidden size
        if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
            target_dim = model.config.hidden_size
        elif hasattr(model, 'config') and hasattr(model.config, 'n_embd'):
            target_dim = model.config.n_embd
        else:
            target_dim = source_dim

        logger.info(f"Detected dimensions: source={source_dim}, target={target_dim}")

        # Apply dimension mapping if needed
        if source_dim != target_dim:
            logger.warning(f"Dimension mismatch detected: {source_dim} → {target_dim}")

            if source_model_id and target_model_id:
                try:
                    from dimension_mapper import get_dimension_mapper
                    mapper = get_dimension_mapper()

                    logger.info(f"Applying dimension mapping: {source_model_id}({source_dim}) → {target_model_id}({target_dim})")
                    persona_vectors = mapper.apply_mapping(
                        persona_vectors,
                        source_model_id,
                        target_model_id,
                        source_dim,
                        target_dim
                    )
                    logger.info("Successfully applied dimension mapping")
                except Exception as e:
                    logger.error(f"Failed to apply dimension mapping: {e}")
                    logger.error(f"Steering will be skipped due to dimension mismatch")
                    return []  # Return empty hooks list
            else:
                logger.error(f"Cannot apply dimension mapping: source_model_id or target_model_id not provided")
                logger.error(f"Steering will be skipped. Provide model IDs to enable cross-model transfer.")
                return []  # Return empty hooks list

    # Only inject at layer 20 (following Chen et al. methodology)
    target_layer_name = "layer_20"

    # Debug: List all available vector layers
    logger.info(f"Available persona vector layers: {list(persona_vectors.keys())}")

    if target_layer_name not in persona_vectors:
        logger.warning(f"No persona vector found for {target_layer_name}, skipping Chen injection")
        # Try to find the closest layer if layer_20 doesn't exist
        available_layers = [k for k in persona_vectors.keys() if k.startswith('layer_')]
        if available_layers:
            # Use middle layer as fallback
            middle_idx = len(available_layers) // 2
            target_layer_name = available_layers[middle_idx]
            logger.info(f"Using fallback layer: {target_layer_name}")
        else:
            return hooks

    try:
        # Extract layer number from target_layer_name (e.g., "layer_20" -> 20)
        layer_num = int(target_layer_name.split('_')[1])
        target_layer = model.model.layers[layer_num]
        logger.info(f"Targeting model layer {layer_num} for Chen injection")
        device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype

        # Prepare persona vector
        persona_vector = persona_vectors[target_layer_name]
        if not isinstance(persona_vector, np.ndarray):
            logger.error(f"Persona vector for {target_layer_name} is not a numpy array")
            return hooks

        vector_tensor = torch.from_numpy(persona_vector).to(dtype=model_dtype, device=device)

        # Debug: Check if vector has meaningful values
        vector_norm = torch.norm(vector_tensor).item()
        vector_mean = torch.mean(vector_tensor).item()
        vector_std = torch.std(vector_tensor).item()
        logger.info(f"Persona vector stats: norm={vector_norm:.6f}, mean={vector_mean:.6f}, std={vector_std:.6f}")

        if vector_norm < 1e-6:
            logger.error(f"Persona vector is essentially zero! norm={vector_norm}")
            return hooks

        scaled_vector = vector_tensor * steering_coefficient
        
        logger.info(f"Chen et al. injection: layer {layer_num}, coefficient {steering_coefficient}, vector shape {scaled_vector.shape}")
        
        # Cache for adapted vector to avoid recomputing on every token
        adapted_vector_cache = {'vector': None, 'target_size': None}
        
        def chen_injection_hook(module, input, output):
            """
            Inject persona vector into layer 20 activations at final token position.
            Following Chen et al. (2024) methodology exactly.
            """
            try:
                logger.info(f"Chen injection hook triggered on layer 20 with coefficient {steering_coefficient}")
                # Handle tuple output (hidden_states, attention_weights, etc.)
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    other_outputs = output[1:]
                else:
                    hidden_states = output
                    other_outputs = ()
                
                # Get dimensions: [batch_size, seq_len, hidden_size]
                if len(hidden_states.shape) != 3:
                    logger.warning(f"Unexpected hidden_states shape: {hidden_states.shape}")
                    return output
                    
                batch_size, seq_len, hidden_size = hidden_states.shape
                
                # Handle vector dimension mismatch (cross-model transfer) - CACHE the adapted vector
                if adapted_vector_cache['vector'] is None or adapted_vector_cache['target_size'] != hidden_size:
                    adapted_vector = scaled_vector
                    if scaled_vector.shape[0] != hidden_size:
                        if not hasattr(chen_injection_hook, '_adaptation_logged'):
                            logger.warning(f"Vector dimension mismatch: {scaled_vector.shape[0]} vs {hidden_size}. Adapting...")
                            chen_injection_hook._adaptation_logged = True
                        
                        if scaled_vector.shape[0] > hidden_size:
                            # Truncate: Use linear interpolation to downsample
                            indices = torch.linspace(0, scaled_vector.shape[0] - 1, hidden_size, device=scaled_vector.device)
                            indices_floor = indices.long()
                            indices_ceil = torch.clamp(indices_floor + 1, max=scaled_vector.shape[0] - 1)
                            weight = (indices - indices_floor.float()).unsqueeze(1)
                            
                            adapted_vector = (1 - weight.squeeze()) * scaled_vector[indices_floor] + weight.squeeze() * scaled_vector[indices_ceil]
                            if not hasattr(chen_injection_hook, '_adaptation_logged'):
                                logger.info(f"Downsampled vector from {scaled_vector.shape[0]} to {hidden_size} using interpolation")
                        else:
                            # Expand: Use linear interpolation to upsample
                            indices = torch.linspace(0, scaled_vector.shape[0] - 1, hidden_size, device=scaled_vector.device)
                            indices_floor = indices.long()
                            indices_ceil = torch.clamp(indices_floor + 1, max=scaled_vector.shape[0] - 1)
                            weight = (indices - indices_floor.float()).unsqueeze(1)
                            
                            adapted_vector = (1 - weight.squeeze()) * scaled_vector[indices_floor] + weight.squeeze() * scaled_vector[indices_ceil]
                            if not hasattr(chen_injection_hook, '_adaptation_logged'):
                                logger.info(f"Upsampled vector from {scaled_vector.shape[0]} to {hidden_size} using interpolation")
                    
                    # Cache the adapted vector
                    adapted_vector_cache['vector'] = adapted_vector
                    adapted_vector_cache['target_size'] = hidden_size
                else:
                    # Use cached adapted vector
                    adapted_vector = adapted_vector_cache['vector']
                
                # Ensure adapted vector is on the same device as hidden_states
                adapted_vector = adapted_vector.to(hidden_states.device)
                
                # Chen et al. style: Apply steering at FINAL token position only
                # Create a new tensor to avoid in-place modification issues
                modified_hidden_states = hidden_states.clone()
                
                # Use moderate internal multiplier - user coefficient controls final strength
                # Sweet spot: coefficient 2-5 gives total injection of 6-15x
                injection_strength = abs(steering_coefficient) * 3.0
                effective_vector = adapted_vector * injection_strength
                
                # Inject only at the LAST token position (Chen et al. methodology)
                modified_hidden_states[:, -1, :] = modified_hidden_states[:, -1, :] + effective_vector.unsqueeze(0).expand(batch_size, -1)
                
                # Debug logging (only on first call to avoid spam)
                if not hasattr(chen_injection_hook, '_debug_logged'):
                    injection_norm = torch.norm(effective_vector).item()
                    logger.info(f"INJECTION DEBUG: effective_vector norm={injection_norm:.6f}, applied to all {seq_len} positions")
                    chen_injection_hook._debug_logged = True
                
                hidden_states = modified_hidden_states
                
                # CRITICAL: Return modified output to replace original layer output
                # The hook MUST return the new output to replace the original
                if isinstance(output, tuple):
                    # Preserve the exact structure of the original output
                    new_output = (hidden_states,) + other_outputs
                    return new_output
                else:
                    return hidden_states
                    
            except Exception as e:
                logger.error(f"Error in Chen injection hook: {e}")
                return output  # Return unmodified output on error
        
        # Register the hook on target layer
        hook_handle = target_layer.register_forward_hook(chen_injection_hook)
        hooks.append(hook_handle)
        
        logger.info(f"Successfully registered Chen et al. injection hook on layer {layer_num}")
        
    except Exception as e:
        logger.error(f"Failed to setup Chen et al. layer 20 injection: {e}")
        return []
    
    return hooks

async def get_model_response(model_id, system_prompt, user_prompt, extract_activations=False, persona_vectors=None, steering_coefficient=0.0):
    """
    Get a response from the specified model with given prompts.
    
    Args:
        model_id: The model identifier
        system_prompt: The system prompt to control model behavior
        user_prompt: The user's question or prompt
        extract_activations: Whether to request activation extraction
    
    Returns:
        A dictionary containing the model's response and activation data if requested
    """
    start_time = time.time()
    
    try:
        # Load model and tokenizer
        model, tokenizer = _load_model(model_id)
        
        # Setup activation extraction if requested
        activations = {}
        hooks = []
        if extract_activations:
            logger.info("Setting up activation extraction hooks...")
            activations, hooks = _attach_activation_hooks(model, model_id)
        
        # Generate response
        logger.info(f"Generating response for: {user_prompt[:50]}...")
        # Increase token limit for GPT-OSS reasoning style and custom trait generation
        if "gpt-oss" in model_id.lower():
            max_tokens = 500
        elif "Generate the complete trait artifacts" in user_prompt:
            # Custom trait generation needs much more tokens for 40 questions
            max_tokens = 4000
        else:
            max_tokens = 200
        
        response_text = _generate_text_response(
            model, tokenizer, model_id, system_prompt, user_prompt,
            max_new_tokens=max_tokens, persona_vectors=persona_vectors,
            steering_coefficient=steering_coefficient
        )
        
        # Process the response
        model_response = {
            "success": True,
            "model_id": model_id,
            "response": response_text,
            "elapsed_time": time.time() - start_time,
        }
        
        # Add activations if requested
        if extract_activations:
            logger.info(f"Captured {len(activations)} activations: {list(activations.keys())}")
            model_response["activations"] = activations
        
        # Remove hooks if added
        if hooks:
            _detach_hooks(hooks)
        
        return model_response
    
    except Exception as e:
        logger.error(f"Error getting model response: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def get_model_info(model_id):
    """Get detailed information about a model."""
    if model_id not in AVAILABLE_MODELS:
        return None
        
    config = AVAILABLE_MODELS[model_id]
    info = {
        "id": model_id,
        "description": config["description"],
        "model_type": config["model_type"],
        "quantization": config.get("quantization", "none"),
        "max_length": config.get("max_length", 2048),
        "loaded": model_id in _loaded_models
    }
    
    if model_id in _loaded_models:
        model = _loaded_models[model_id]
        info["memory_usage"] = f"~{torch.cuda.memory_allocated() / 1024**3:.1f}GB"
        info["device"] = str(model.device) if hasattr(model, 'device') else "unknown"
    
    return info

def unload_model(model_id):
    """Unload a model to free memory."""
    if model_id in _loaded_models:
        del _loaded_models[model_id]
        del _loaded_tokenizers[model_id]
        if model_id in _model_configs:
            del _model_configs[model_id]

        # Force garbage collection
        import gc
        gc.collect()

        # Clear GPU cache (CUDA or MPS)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            # Clear MPS cache for Apple Silicon
            gc.collect()

        logger.info(f"Unloaded model {model_id}")
        return True

    return False