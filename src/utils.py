"""Core utilities for deterministic behavior and device management."""

import os
import random
from typing import Optional, Union
import numpy as np
import tensorflow as tf
import torch


def set_deterministic_seed(seed: int = 42) -> None:
    """Set deterministic seeds for all random number generators.
    
    Args:
        seed: Random seed value for reproducibility.
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # TensorFlow
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_type: Optional[str] = None) -> str:
    """Get available device with fallback to CPU.
    
    Args:
        device_type: Preferred device type ('cuda', 'cpu', 'mps').
        
    Returns:
        Available device string.
    """
    if device_type is None:
        if tf.config.list_physical_devices('GPU'):
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    if device_type == 'cuda' and not tf.config.list_physical_devices('GPU'):
        print("CUDA not available, falling back to CPU")
        return 'cpu'
    
    return device_type


def configure_tensorflow_memory_growth() -> None:
    """Configure TensorFlow to use memory growth for GPU."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU memory growth configuration failed: {e}")


def get_model_size_mb(model: Union[tf.keras.Model, torch.nn.Module]) -> float:
    """Calculate model size in MB.
    
    Args:
        model: TensorFlow or PyTorch model.
        
    Returns:
        Model size in megabytes.
    """
    if isinstance(model, tf.keras.Model):
        # TensorFlow model
        model_size = 0
        for layer in model.layers:
            for weight in layer.weights:
                model_size += np.prod(weight.shape) * 4  # float32 = 4 bytes
        return model_size / (1024 * 1024)  # Convert to MB
    else:
        # PyTorch model
        param_size = 0
        buffer_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return (param_size + buffer_size) / (1024 * 1024)  # Convert to MB
