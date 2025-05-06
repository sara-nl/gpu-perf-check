"""
NCCL bandwidth benchmarking.

This module provides classes and functions for benchmarking NCCL collective operations
bandwidth with support for different operations and multi-node execution.
"""
import torch
import torch.distributed as dist
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import os
import time
from functools import partial

from ..base import BenchmarkTest, BenchmarkResult, TestFunctionRegistry
from ..utils.profiling import profile_op, calculate_bandwidth
from ..utils.device import get_device_info, get_free_memory, sanitize_filename, is_slurm_environment, get_slurm_info
from ..utils.plotting import plot_bandwidth_latency
from .operations import create_nccl_op, NCCL_OPERATIONS, get_data_transferred_bytes


class NCCLBandwidthTest(BenchmarkTest):
    """
    Benchmark for NCCL bandwidth using various collective operations.
    """
    def __init__(
        self,
        name: str = "nccl_bandwidth",
        device_id: int = 0,
        output_dir: Optional[str] = None,
        operations: Optional[List[str]] = None,
        max_memory_fraction: float = 0.45,
        runs: int = 10,
        n_warmup: int = 5,
        verbose: bool = True,
        distributed_init: bool = True
    ):
        """
        Initialize an NCCL bandwidth benchmark.
        
        Args:
            name: Name of the benchmark
            device_id: CUDA device ID to use
            output_dir: Directory to save results
            operations: List of NCCL operations to benchmark (default: ['all_reduce', 'all_gather', 'broadcast'])
            max_memory_fraction: Maximum fraction of free memory to use
            runs: Number of runs per size
            n_warmup: Number of warmup runs
            verbose: Whether to print progress information
            distributed_init: Whether to initialize the distributed environment
        """
        super().__init__(name, device_id, output_dir)
        
        self.operations = operations or ['all_reduce', 'all_gather', 'broadcast']
        self.max_memory_fraction = max_memory_fraction
        self.runs = runs
        self.n_warmup = n_warmup
        self.verbose = verbose
        
        # Initialize distributed environment if needed
        self.distributed_init = distributed_init
        if distributed_init and not dist.is_initialized():
            self._init_distributed()
        
        # Validate operations
        for op in self.operations:
            if op not in NCCL_OPERATIONS:
                raise ValueError(f"Unknown NCCL operation: {op}. Available operations: {list(NCCL_OPERATIONS.keys())}")
        
        # Create registry for operations
        self.registry = TestFunctionRegistry()
        for op in self.operations:
            self.registry.register(
                name=op,
                function=lambda op_name=op: create_nccl_op(op_name, **NCCL_OPERATIONS[op_name]['parameters']),
                description=NCCL_OPERATIONS[op]['description'],
                parameters=NCCL_OPERATIONS[op]['parameters']
            )
    
    def _init_distributed(self) -> None:
        """Initialize the distributed environment."""
        # Check if running in SLURM
        if is_slurm_environment():
            # Get SLURM environment variables
            slurm_info = get_slurm_info()
            
            # Set environment variables for PyTorch distributed
            os.environ['MASTER_ADDR'] = os.environ.get('SLURM_LAUNCH_NODE_IPADDR', 'localhost')
            os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
            os.environ['RANK'] = slurm_info.get('SLURM_PROCID', '0')
            os.environ['WORLD_SIZE'] = slurm_info.get('SLURM_NTASKS', '1')
            
            # Set local rank based on SLURM_LOCALID
            local_rank = int(slurm_info.get('SLURM_LOCALID', '0'))
            torch.cuda.set_device(local_rank)
            
            if self.verbose and int(os.environ['RANK']) == 0:
                print(f"Initializing distributed environment with SLURM:")
                print(f"  MASTER_ADDR: {os.environ['MASTER_ADDR']}")
                print(f"  MASTER_PORT: {os.environ['MASTER_PORT']}")
                print(f"  RANK: {os.environ['RANK']}")
                print(f"  WORLD_SIZE: {os.environ['WORLD_SIZE']}")
                print(f"  LOCAL_RANK: {local_rank}")
        else:
            # Single-node setup
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '29500'
            os.environ['RANK'] = '0'
            os.environ['WORLD_SIZE'] = '1'
            local_rank = 0
            torch.cuda.set_device(local_rank)
        
        # Initialize the process group
        dist.init_process_group(backend='nccl')
        
        if self.verbose and dist.get_rank() == 0:
            print(f"Distributed environment initialized:")
            print(f"  World size: {dist.get_world_size()}")
            print(f"  Rank: {dist.get_rank()}")
    
    def generate_sizes(self, min_bytes: int = 1, max_bytes: Optional[int] = None) -> List[int]:
        """
        Generate a list of sizes to benchmark.
        
        Args:
            min_bytes: Minimum size in bytes
            max_bytes: Maximum size in bytes (default: calculated based on free memory)
            
        Returns:
            List of sizes in bytes
        """
        if max_bytes is None:
            free_mem = get_free_memory(self.device_id)
            max_bytes = int(free_mem * self.max_memory_fraction)
        
        # Generate powers of 2
        sizes = []
        size = min_bytes
        while size <= max_bytes:
            sizes.append(size)
            size *= 2
        
        return sizes
    
    def measure_bandwidth(
        self,
        op_name: str,
        size_bytes: int
    ) -> Dict[str, float]:
        """
        Measure bandwidth for a specific operation and size.
        
        Args:
            op_name: Name of the operation
            size_bytes: Size in bytes
            
        Returns:
            Dictionary with bandwidth measurement results
        """
        # Get the operation and setup functions
        op_fn, setup_fn = self.registry.get(op_name)()
        
        # Create a wrapper for the setup function that takes size_bytes and device
        def wrapped_setup():
            args, _ = setup_fn(size_bytes, self.device)
            return args
        
        # Create a wrapper for the operation function
        def wrapped_op():
            args = wrapped_setup()
            if isinstance(args, tuple):
                op_fn(*args)
            else:
                op_fn(args)
            torch.cuda.synchronize()
        
        # Profile the operation
        stats = profile_op(
            op_fn=wrapped_op,
            n_warmup=self.n_warmup,
            runs=self.runs
        )
        
        # Calculate bandwidth based on the amount of data transferred
        data_transferred = get_data_transferred_bytes(op_name, size_bytes)
        bandwidth = calculate_bandwidth(data_transferred, stats['mean'])
        
        if self.verbose and dist.get_rank() == 0:
            print(f"{op_name} size={size_bytes} bytes: {stats['mean']*1000:.3f} ms, {bandwidth/1e9:.2f} GB/s")
        
        return {
            'op_name': op_name,
            'size_bytes': size_bytes,
            'latency_mean': stats['mean'],
            'latency_std': stats['std'],
            'bandwidth_mean': bandwidth,
            'bandwidth_std': bandwidth * (stats['std'] / stats['mean']) if stats['mean'] > 0 else 0,
            'data_transferred': data_transferred
        }
    
    def run(
        self,
        sizes_bytes: Optional[List[int]] = None,
        operations: Optional[List[str]] = None
    ) -> BenchmarkResult:
        """
        Run the benchmark and return results.
        
        Args:
            sizes_bytes: List of sizes to benchmark (default: generated automatically)
            operations: List of operations to benchmark (default: all registered operations)
            
        Returns:
            BenchmarkResult containing the benchmark results
        """
        # Use default sizes if not provided
        if sizes_bytes is None:
            sizes_bytes = self.generate_sizes()
        
        # Use default operations if not provided
        operations = operations or self.operations
        
        # Get device information
        device_info = get_device_info(self.device_id)
        
        # Add distributed information to device info
        device_info['world_size'] = dist.get_world_size()
        device_info['rank'] = dist.get_rank()
        
        # Add SLURM information if available
        if is_slurm_environment():
            device_info['slurm'] = get_slurm_info()
        
        # Initialize results
        results = {op: {
            'size_bytes': [],
            'latency_mean': [],
            'latency_std': [],
            'bandwidth_mean': [],
            'bandwidth_std': [],
            'data_transferred': []
        } for op in operations}
        
        # Run benchmarks
        for op in operations:
            if self.verbose and dist.get_rank() == 0:
                print(f"\nBenchmarking {op} operation...")
            
            for size in sizes_bytes:
                try:
                    measurement = self.measure_bandwidth(op, size)
                    
                    results[op]['size_bytes'].append(size)
                    results[op]['latency_mean'].append(measurement['latency_mean'])
                    results[op]['latency_std'].append(measurement['latency_std'])
                    results[op]['bandwidth_mean'].append(measurement['bandwidth_mean'])
                    results[op]['bandwidth_std'].append(measurement['bandwidth_std'])
                    results[op]['data_transferred'].append(measurement['data_transferred'])
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        if self.verbose and dist.get_rank() == 0:
                            print(f"Out of memory at size {size} bytes, stopping benchmark for {op}")
                        break
                    else:
                        if self.verbose and dist.get_rank() == 0:
                            print(f"Error at size {size} bytes for {op}: {e}")
                        continue
        
        # Create benchmark result
        parameters = {
            'operations': operations,
            'sizes_bytes': sizes_bytes,
            'runs': self.runs,
            'n_warmup': self.n_warmup,
            'world_size': dist.get_world_size()
        }
        
        return BenchmarkResult(
            name=self.name,
            device_info=device_info,
            parameters=parameters,
            results=results
        )
    
    def plot(
        self,
        result: BenchmarkResult,
        output_file: Optional[str] = None,
        plot_latency: bool = True,
        plot_bandwidth: bool = True,
        operations: Optional[List[str]] = None
    ) -> None:
        """
        Plot the benchmark results.
        
        Args:
            result: BenchmarkResult to plot
            output_file: Output file path (default: auto-generated)
            plot_latency: Whether to plot latency
            plot_bandwidth: Whether to plot bandwidth
            operations: List of operations to include in the plot (default: all)
        """
        # Only the root process should plot
        if dist.get_rank() != 0:
            return
        
        if output_file is None:
            device_name = sanitize_filename(result.device_info.get('name', 'unknown'))
            world_size = result.device_info.get('world_size', 1)
            output_file = os.path.join(self.output_dir, f"{self.name}_{device_name}_n{world_size}.png")
        
        # Filter operations if specified
        results = result.results
        if operations:
            results = {op: data for op, data in results.items() if op in operations}
        
        # Create plot title
        device_name = result.device_info.get('name', 'Unknown GPU')
        world_size = result.device_info.get('world_size', 1)
        title = f"NCCL {'Bandwidth' if plot_bandwidth else ''}{' & ' if plot_bandwidth and plot_latency else ''}{'Latency' if plot_latency else ''} ({device_name}, {world_size} processes)"
        
        # Create plot
        plot_bandwidth_latency(
            results=results,
            title=title,
            output_file=output_file,
            include_bandwidth=plot_bandwidth,
            include_latency=plot_latency
        )
        
        if self.verbose:
            print(f"Plot saved to {output_file}")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.distributed_init and dist.is_initialized():
            dist.destroy_process_group()


def main(
    device_id: int = 0,
    operations: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    runs: int = 10,
    n_warmup: int = 5,
    max_memory_fraction: float = 0.45,
    verbose: bool = True,
    output_file: Optional[str] = None
) -> Optional[BenchmarkResult]:
    """
    Run an NCCL bandwidth benchmark and plot the results.
    
    Args:
        device_id: CUDA device ID to use
        operations: List of NCCL operations to benchmark (default: ['all_reduce', 'all_gather', 'broadcast'])
        output_dir: Directory to save results
        runs: Number of runs per size
        n_warmup: Number of warmup runs
        max_memory_fraction: Maximum fraction of free memory to use
        verbose: Whether to print progress information
        output_file: Output file path for the plot (default: auto-generated)
        
    Returns:
        BenchmarkResult containing the benchmark results (only for rank 0)
    """
    # Create benchmark
    benchmark = NCCLBandwidthTest(
        device_id=device_id,
        operations=operations,
        output_dir=output_dir,
        runs=runs,
        n_warmup=n_warmup,
        max_memory_fraction=max_memory_fraction,
        verbose=verbose
    )
    
    try:
        # Run benchmark
        result = benchmark.run()
        
        # Plot and save results (only on rank 0)
        if dist.get_rank() == 0:
            benchmark.plot(result, output_file=output_file)
            benchmark.save_results(result)
            return result
        return None
    finally:
        # Clean up
        benchmark.cleanup()
