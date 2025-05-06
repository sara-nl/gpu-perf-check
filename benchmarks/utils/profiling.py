"""
Profiling utilities for GPU operations.

This module provides tools for profiling GPU operations with proper synchronization
and statistical analysis.
"""
import torch
import numpy as np
from typing import Callable, Dict, List, Optional, Any, Union


def profile_op(
    op_fn: Callable[[], None],
    setup_fn: Optional[Callable[[], Any]] = None,
    n_warmup: int = 5,
    runs: int = 5,
) -> Dict[str, Union[float, List[float]]]:
    """
    Generic GPU profiling utility.
    
    Args:
        op_fn: Function that performs the operation to time
        setup_fn: Function to set up and return any resources/inputs (e.g. tensors)
        n_warmup: Number of warmup runs to perform
        runs: Number of timed runs to perform
        
    Returns:
        Dict with 'mean', 'std', 'latencies' (in seconds), and 'runs'
    """
    # Warmup phase
    for _ in range(n_warmup):
        if setup_fn:
            setup_fn()
        op_fn()
        torch.cuda.synchronize()

    # Measurement phase
    latencies = []
    for _ in range(runs):
        if setup_fn:
            setup_fn()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        op_fn()
        end_evt.record()
        torch.cuda.synchronize()
        latencies.append(start_evt.elapsed_time(end_evt) / 1e3)  # Convert ms to seconds

    latencies_array = np.array(latencies)
    return {
        'mean': float(latencies_array.mean()),
        'std': float(latencies_array.std()),
        'latencies': latencies_array.tolist(),
        'runs': runs
    }


def calculate_bandwidth(size_bytes: int, time_seconds: float) -> float:
    """
    Calculate bandwidth in GB/s given size and time.
    
    Args:
        size_bytes: Size of data transferred in bytes
        time_seconds: Time taken in seconds
        
    Returns:
        Bandwidth in GB/s (10^9 bytes per second)
    """
    if time_seconds <= 0:
        return 0.0
    return (size_bytes / 1e9) / time_seconds  # GB/s


def calculate_bandwidth_gib(size_bytes: int, time_seconds: float) -> float:
    """
    Calculate bandwidth in GiB/s given size and time.
    
    Args:
        size_bytes: Size of data transferred in bytes
        time_seconds: Time taken in seconds
        
    Returns:
        Bandwidth in GiB/s (2^30 bytes per second)
    """
    if time_seconds <= 0:
        return 0.0
    return (size_bytes / (2**30)) / time_seconds  # GiB/s
