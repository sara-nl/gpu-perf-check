"""
Device utilities for GPU benchmarking.

This module provides functions for querying device properties and managing CUDA devices.
"""
import torch
import os
import re
from typing import Dict, List, Optional, Union, Tuple, Any


def get_device_info(device_id: int) -> Dict[str, Any]:
    """
    Get detailed information about a CUDA device.
    
    Args:
        device_id: CUDA device ID
        
    Returns:
        Dictionary with device properties
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    if device_id >= torch.cuda.device_count():
        raise ValueError(f"Device ID {device_id} is out of range. Available devices: {torch.cuda.device_count()}")
    
    props = torch.cuda.get_device_properties(device_id)
    breakpoint()
    return {
        'name': props.name,
        'total_memory': props.total_memory,
        'major': props.major,
        'minor': props.minor,
        # 'multi_processor_count': props.multi_processor_count,
        # 'max_threads_per_block': props.max_threads_per_block,
        # 'max_threads_per_multiprocessor': props.max_threads_per_multiprocessor,
        # 'warp_size': props.warp_size,
        # 'compute_capability': f"{props.major}.{props.minor}"
    }


def get_free_memory(device_id: int) -> int:
    """
    Get free memory on a CUDA device in bytes.
    
    Args:
        device_id: CUDA device ID
        
    Returns:
        Free memory in bytes
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    if device_id >= torch.cuda.device_count():
        raise ValueError(f"Device ID {device_id} is out of range. Available devices: {torch.cuda.device_count()}")
    
    return torch.cuda.mem_get_info(device_id)[0]  # Returns (free, total)


def get_memory_utilization(device_id: int) -> float:
    """
    Get memory utilization percentage on a CUDA device.
    
    Args:
        device_id: CUDA device ID
        
    Returns:
        Memory utilization as a percentage (0-100)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    if device_id >= torch.cuda.device_count():
        raise ValueError(f"Device ID {device_id} is out of range. Available devices: {torch.cuda.device_count()}")
    
    free, total = torch.cuda.mem_get_info(device_id)
    return 100.0 * (1.0 - free / total)


def is_slurm_environment() -> bool:
    """
    Check if running in a SLURM environment.
    
    Returns:
        True if running in a SLURM environment, False otherwise
    """
    return 'SLURM_JOB_ID' in os.environ


def get_slurm_info() -> Dict[str, str]:
    """
    Get information about the current SLURM job.
    
    Returns:
        Dictionary with SLURM environment variables
    """
    if not is_slurm_environment():
        return {}
    
    slurm_vars = [
        'SLURM_JOB_ID',
        'SLURM_JOB_NODELIST',
        'SLURM_NTASKS',
        'SLURM_NTASKS_PER_NODE',
        'SLURM_GPUS_PER_NODE',
        'SLURM_JOB_CPUS_PER_NODE',
        'SLURM_PROCID',
        'SLURM_LOCALID',
        'SLURM_NODEID'
    ]
    
    return {var: os.environ.get(var, '') for var in slurm_vars if var in os.environ}


def sanitize_filename(name: str) -> str:
    """
    Sanitize a string to be used as a filename.
    
    Args:
        name: Input string
        
    Returns:
        Sanitized string suitable for use as a filename
    """
    return re.sub(r'[^\w\-]+', '-', name)
