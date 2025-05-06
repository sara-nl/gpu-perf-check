"""
GPU memory operations for benchmarking.

This module provides various memory operations (read, write, copy, fill, etc.)
that can be used for benchmarking GPU memory bandwidth and latency.
"""
import torch
from typing import Callable, Dict, List, Optional, Any, Tuple
from functools import partial


def zero_tensor(tensor: torch.Tensor) -> None:
    """
    Fill a tensor with zeros.
    
    Args:
        tensor: The tensor to fill with zeros
    """
    tensor.zero_()


def fill_tensor(tensor: torch.Tensor, value: float = 1.0) -> None:
    """
    Fill a tensor with a specific value.
    
    Args:
        tensor: The tensor to fill
        value: Value to fill the tensor with
    """
    tensor.fill_(value)


def copy_tensor(src: torch.Tensor, dst: torch.Tensor) -> None:
    """
    Copy data from source tensor to destination tensor.
    
    Args:
        src: Source tensor
        dst: Destination tensor
    """
    dst.copy_(src)


def add_tensors(a: torch.Tensor, b: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Add two tensors.
    
    Args:
        a: First tensor
        b: Second tensor
        out: Optional output tensor
        
    Returns:
        Result tensor
    """
    if out is not None:
        torch.add(a, b, out=out)
        return out
    return torch.add(a, b)


def mul_tensors(a: torch.Tensor, b: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Multiply two tensors element-wise.
    
    Args:
        a: First tensor
        b: Second tensor
        out: Optional output tensor
        
    Returns:
        Result tensor
    """
    if out is not None:
        torch.mul(a, b, out=out)
        return out
    return torch.mul(a, b)


def create_memory_op(op_name: str, **kwargs) -> Tuple[Callable, Dict[str, Any]]:
    """
    Create a memory operation function based on the operation name.
    
    Args:
        op_name: Name of the operation ('zero', 'fill', 'copy', 'add', 'mul')
        **kwargs: Additional parameters for the operation
        
    Returns:
        Tuple of (operation function, setup function)
    """
    if op_name == 'zero':
        def setup_fn(size_bytes: int, device: torch.device) -> Tuple[torch.Tensor, Dict]:
            # Convert bytes to number of elements (assuming float32)
            num_elements = size_bytes // 4
            tensor = torch.empty(num_elements, dtype=torch.float32, device=device)
            return tensor, {'tensor': tensor}
        
        def op_fn(tensor: torch.Tensor) -> None:
            zero_tensor(tensor)
        
        return op_fn, setup_fn
    
    elif op_name == 'fill':
        value = kwargs.get('value', 1.0)
        
        def setup_fn(size_bytes: int, device: torch.device) -> Tuple[torch.Tensor, Dict]:
            num_elements = size_bytes // 4
            tensor = torch.empty(num_elements, dtype=torch.float32, device=device)
            return tensor, {'tensor': tensor}
        
        def op_fn(tensor: torch.Tensor) -> None:
            fill_tensor(tensor, value)
        
        return op_fn, setup_fn
    
    elif op_name == 'copy':
        def setup_fn(size_bytes: int, device: torch.device) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict]:
            num_elements = size_bytes // 4
            src = torch.ones(num_elements, dtype=torch.float32, device=device)
            dst = torch.empty(num_elements, dtype=torch.float32, device=device)
            return (src, dst), {'src': src, 'dst': dst}
        
        def op_fn(src: torch.Tensor, dst: torch.Tensor) -> None:
            copy_tensor(src, dst)
        
        return op_fn, setup_fn
    
    elif op_name == 'add':
        def setup_fn(size_bytes: int, device: torch.device) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict]:
            num_elements = size_bytes // 4
            a = torch.ones(num_elements, dtype=torch.float32, device=device)
            b = torch.ones(num_elements, dtype=torch.float32, device=device)
            out = torch.empty(num_elements, dtype=torch.float32, device=device)
            return (a, b, out), {'a': a, 'b': b, 'out': out}
        
        def op_fn(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
            add_tensors(a, b, out)
        
        return op_fn, setup_fn
    
    elif op_name == 'mul':
        def setup_fn(size_bytes: int, device: torch.device) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict]:
            num_elements = size_bytes // 4
            a = torch.ones(num_elements, dtype=torch.float32, device=device)
            b = torch.ones(num_elements, dtype=torch.float32, device=device)
            out = torch.empty(num_elements, dtype=torch.float32, device=device)
            return (a, b, out), {'a': a, 'b': b, 'out': out}
        
        def op_fn(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
            mul_tensors(a, b, out)
        
        return op_fn, setup_fn
    
    else:
        raise ValueError(f"Unknown memory operation: {op_name}")


# Registry of available memory operations
MEMORY_OPERATIONS = {
    'zero': {
        'description': 'Fill tensor with zeros',
        'parameters': {}
    },
    'fill': {
        'description': 'Fill tensor with a specific value',
        'parameters': {'value': 1.0}
    },
    'copy': {
        'description': 'Copy data from source tensor to destination tensor',
        'parameters': {}
    },
    'add': {
        'description': 'Add two tensors',
        'parameters': {}
    },
    'mul': {
        'description': 'Multiply two tensors element-wise',
        'parameters': {}
    }
}
