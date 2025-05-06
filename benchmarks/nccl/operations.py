"""
NCCL collective operations for benchmarking.

This module provides various NCCL collective operations (all_reduce, all_gather, etc.)
that can be used for benchmarking NCCL bandwidth and latency.
"""
import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Union, Any, Tuple, Callable


def all_reduce(tensor: torch.Tensor, group: Optional[dist.ProcessGroup] = None) -> None:
    """
    Perform an all-reduce operation.
    
    Args:
        tensor: The tensor to reduce
        group: Optional process group
    """
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)


def all_gather(tensor: torch.Tensor, group: Optional[dist.ProcessGroup] = None) -> List[torch.Tensor]:
    """
    Perform an all-gather operation.
    
    Args:
        tensor: The tensor to gather
        group: Optional process group
        
    Returns:
        List of gathered tensors
    """
    world_size = dist.get_world_size(group)
    output_tensors = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(output_tensors, tensor, group=group)
    return output_tensors


def broadcast(tensor: torch.Tensor, src: int = 0, group: Optional[dist.ProcessGroup] = None) -> None:
    """
    Perform a broadcast operation.
    
    Args:
        tensor: The tensor to broadcast
        src: Source rank
        group: Optional process group
    """
    dist.broadcast(tensor, src=src, group=group)


def reduce(tensor: torch.Tensor, dst: int = 0, group: Optional[dist.ProcessGroup] = None) -> None:
    """
    Perform a reduce operation.
    
    Args:
        tensor: The tensor to reduce
        dst: Destination rank
        group: Optional process group
    """
    dist.reduce(tensor, dst=dst, op=dist.ReduceOp.SUM, group=group)


def reduce_scatter(tensor: torch.Tensor, input_tensors: List[torch.Tensor], group: Optional[dist.ProcessGroup] = None) -> None:
    """
    Perform a reduce-scatter operation.
    
    Args:
        tensor: Output tensor
        input_tensors: List of input tensors
        group: Optional process group
    """
    dist.reduce_scatter(tensor, input_tensors, op=dist.ReduceOp.SUM, group=group)


def all_to_all(output_tensors: List[torch.Tensor], input_tensors: List[torch.Tensor], group: Optional[dist.ProcessGroup] = None) -> None:
    """
    Perform an all-to-all operation.
    
    Args:
        output_tensors: List of output tensors
        input_tensors: List of input tensors
        group: Optional process group
    """
    dist.all_to_all(output_tensors, input_tensors, group=group)


def create_nccl_op(op_name: str, **kwargs) -> Tuple[Callable, Callable]:
    """
    Create an NCCL operation function based on the operation name.
    
    Args:
        op_name: Name of the operation ('all_reduce', 'all_gather', etc.)
        **kwargs: Additional parameters for the operation
        
    Returns:
        Tuple of (operation function, setup function)
    """
    world_size = dist.get_world_size()
    
    if op_name == 'all_reduce':
        def setup_fn(size_bytes: int, device: torch.device) -> Tuple[torch.Tensor, Dict]:
            # Convert bytes to number of elements (assuming float32)
            num_elements = size_bytes // 4
            tensor = torch.ones(num_elements, dtype=torch.float32, device=device)
            return tensor, {'tensor': tensor}
        
        def op_fn(tensor: torch.Tensor) -> None:
            all_reduce(tensor)
        
        return op_fn, setup_fn
    
    elif op_name == 'all_gather':
        def setup_fn(size_bytes: int, device: torch.device) -> Tuple[torch.Tensor, Dict]:
            # For all_gather, the total size is world_size * size_bytes
            # So we divide by world_size to get the correct input tensor size
            num_elements = (size_bytes // 4) // world_size
            tensor = torch.ones(num_elements, dtype=torch.float32, device=device)
            return tensor, {'tensor': tensor}
        
        def op_fn(tensor: torch.Tensor) -> None:
            all_gather(tensor)
        
        return op_fn, setup_fn
    
    elif op_name == 'broadcast':
        src = kwargs.get('src', 0)
        
        def setup_fn(size_bytes: int, device: torch.device) -> Tuple[torch.Tensor, Dict]:
            num_elements = size_bytes // 4
            tensor = torch.ones(num_elements, dtype=torch.float32, device=device)
            return tensor, {'tensor': tensor, 'src': src}
        
        def op_fn(tensor: torch.Tensor) -> None:
            broadcast(tensor, src=src)
        
        return op_fn, setup_fn
    
    elif op_name == 'reduce':
        dst = kwargs.get('dst', 0)
        
        def setup_fn(size_bytes: int, device: torch.device) -> Tuple[torch.Tensor, Dict]:
            num_elements = size_bytes // 4
            tensor = torch.ones(num_elements, dtype=torch.float32, device=device)
            return tensor, {'tensor': tensor, 'dst': dst}
        
        def op_fn(tensor: torch.Tensor) -> None:
            reduce(tensor, dst=dst)
        
        return op_fn, setup_fn
    
    elif op_name == 'reduce_scatter':
        def setup_fn(size_bytes: int, device: torch.device) -> Tuple[Tuple[torch.Tensor, List[torch.Tensor]], Dict]:
            # For reduce_scatter, the input is world_size tensors of size_bytes each
            # The output is a single tensor of size_bytes
            num_elements = size_bytes // 4
            output_tensor = torch.empty(num_elements, dtype=torch.float32, device=device)
            
            # Create world_size input tensors
            input_tensors = [torch.ones(num_elements, dtype=torch.float32, device=device) 
                            for _ in range(world_size)]
            
            return (output_tensor, input_tensors), {'output_tensor': output_tensor, 'input_tensors': input_tensors}
        
        def op_fn(output_tensor: torch.Tensor, input_tensors: List[torch.Tensor]) -> None:
            reduce_scatter(output_tensor, input_tensors)
        
        return op_fn, setup_fn
    
    elif op_name == 'all_to_all':
        def setup_fn(size_bytes: int, device: torch.device) -> Tuple[Tuple[List[torch.Tensor], List[torch.Tensor]], Dict]:
            # For all_to_all, both input and output are lists of world_size tensors
            num_elements = (size_bytes // 4) // world_size
            
            output_tensors = [torch.empty(num_elements, dtype=torch.float32, device=device) 
                             for _ in range(world_size)]
            
            input_tensors = [torch.ones(num_elements, dtype=torch.float32, device=device) 
                            for _ in range(world_size)]
            
            return (output_tensors, input_tensors), {'output_tensors': output_tensors, 'input_tensors': input_tensors}
        
        def op_fn(output_tensors: List[torch.Tensor], input_tensors: List[torch.Tensor]) -> None:
            all_to_all(output_tensors, input_tensors)
        
        return op_fn, setup_fn
    
    else:
        raise ValueError(f"Unknown NCCL operation: {op_name}")


# Registry of available NCCL operations
NCCL_OPERATIONS = {
    'all_reduce': {
        'description': 'All-reduce operation (sum)',
        'parameters': {}
    },
    'all_gather': {
        'description': 'All-gather operation',
        'parameters': {}
    },
    'broadcast': {
        'description': 'Broadcast operation',
        'parameters': {'src': 0}
    },
    'reduce': {
        'description': 'Reduce operation (sum)',
        'parameters': {'dst': 0}
    },
    'reduce_scatter': {
        'description': 'Reduce-scatter operation (sum)',
        'parameters': {}
    },
    'all_to_all': {
        'description': 'All-to-all operation',
        'parameters': {}
    }
}


# Function to calculate the amount of data transferred for each operation
def get_data_transferred_bytes(op_name: str, tensor_size_bytes: int) -> int:
    """
    Calculate the amount of data transferred for a given NCCL operation.
    
    Args:
        op_name: Name of the operation
        tensor_size_bytes: Size of the tensor in bytes
        
    Returns:
        Amount of data transferred in bytes
    """
    world_size = dist.get_world_size()
    
    if op_name == 'all_reduce':
        # Each rank sends and receives from all other ranks
        return 2 * tensor_size_bytes * (world_size - 1)
    
    elif op_name == 'all_gather':
        # Each rank sends its data to all other ranks
        return tensor_size_bytes * (world_size - 1)
    
    elif op_name == 'broadcast':
        # One rank sends to all other ranks
        return tensor_size_bytes * (world_size - 1)
    
    elif op_name == 'reduce':
        # All ranks send to one rank
        return tensor_size_bytes * (world_size - 1)
    
    elif op_name == 'reduce_scatter':
        # Each rank sends a portion to all other ranks
        return tensor_size_bytes * (world_size - 1)
    
    elif op_name == 'all_to_all':
        # Each rank sends a different portion to each other rank
        return tensor_size_bytes * (world_size - 1)
    
    else:
        raise ValueError(f"Unknown NCCL operation: {op_name}")
