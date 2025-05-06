"""
GPU memory latency benchmarking.

This module provides classes and functions for benchmarking GPU memory latency
with support for different memory operations.
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import os
from functools import partial

from ..base import BenchmarkTest, BenchmarkResult, TestFunctionRegistry
from ..utils.profiling import profile_op
from ..utils.device import get_device_info, get_free_memory, sanitize_filename
from ..utils.plotting import plot_bandwidth_latency
from .operations import create_memory_op, MEMORY_OPERATIONS


class MemoryLatencyTest(BenchmarkTest):
    """
    Benchmark for GPU memory latency using various memory operations.
    """
    def __init__(
        self,
        name: str = "memory_latency",
        device_id: int = 0,
        output_dir: Optional[str] = None,
        operations: Optional[List[str]] = None,
        max_memory_fraction: float = 0.45,
        runs: int = 10,
        n_warmup: int = 5,
        verbose: bool = True
    ):
        """
        Initialize a memory latency benchmark.
        
        Args:
            name: Name of the benchmark
            device_id: CUDA device ID to use
            output_dir: Directory to save results
            operations: List of memory operations to benchmark (default: ['zero', 'fill', 'copy'])
            max_memory_fraction: Maximum fraction of free memory to use
            runs: Number of runs per size
            n_warmup: Number of warmup runs
            verbose: Whether to print progress information
        """
        super().__init__(name, device_id, output_dir)
        
        self.operations = operations or ['zero', 'fill', 'copy']
        self.max_memory_fraction = max_memory_fraction
        self.runs = runs
        self.n_warmup = n_warmup
        self.verbose = verbose
        
        # Validate operations
        for op in self.operations:
            if op not in MEMORY_OPERATIONS:
                raise ValueError(f"Unknown memory operation: {op}. Available operations: {list(MEMORY_OPERATIONS.keys())}")
        
        # Create registry for operations
        self.registry = TestFunctionRegistry()
        for op in self.operations:
            self.registry.register(
                name=op,
                function=lambda op_name=op: create_memory_op(op_name, **MEMORY_OPERATIONS[op_name]['parameters']),
                description=MEMORY_OPERATIONS[op]['description'],
                parameters=MEMORY_OPERATIONS[op]['parameters']
            )
    
    def generate_sizes(self, min_bytes: int = 1, max_bytes: Optional[int] = None) -> List[int]:
        """
        Generate a list of sizes to benchmark, focusing on small sizes for latency testing.
        
        Args:
            min_bytes: Minimum size in bytes
            max_bytes: Maximum size in bytes (default: calculated based on free memory)
            
        Returns:
            List of sizes in bytes
        """
        if max_bytes is None:
            free_mem = get_free_memory(self.device_id)
            max_bytes = int(free_mem * self.max_memory_fraction)
            # For latency testing, we don't need to go as high as for bandwidth
            max_bytes = min(max_bytes, 1024 * 1024 * 64)  # 64 MB max
        
        # Generate powers of 2, with more granularity at smaller sizes
        sizes = []
        size = min_bytes
        while size <= max_bytes:
            sizes.append(size)
            # Use smaller step sizes for small tensors to better capture latency characteristics
            if size < 4096:
                size *= 2
            else:
                size *= 4
        
        return sizes
    
    def measure_latency(
        self,
        op_name: str,
        size_bytes: int
    ) -> Dict[str, float]:
        """
        Measure latency for a specific operation and size.
        
        Args:
            op_name: Name of the operation
            size_bytes: Size in bytes
            
        Returns:
            Dictionary with latency measurement results
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
        
        if self.verbose:
            print(f"{op_name} size={size_bytes} bytes: {stats['mean']*1000:.3f} ms")
        
        return {
            'op_name': op_name,
            'size_bytes': size_bytes,
            'latency_mean': stats['mean'],
            'latency_std': stats['std']
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
        
        # Initialize results
        results = {op: {
            'size_bytes': [],
            'latency_mean': [],
            'latency_std': []
        } for op in operations}
        
        # Run benchmarks
        for op in operations:
            if self.verbose:
                print(f"\nBenchmarking {op} operation...")
            
            for size in sizes_bytes:
                try:
                    measurement = self.measure_latency(op, size)
                    
                    results[op]['size_bytes'].append(size)
                    results[op]['latency_mean'].append(measurement['latency_mean'])
                    results[op]['latency_std'].append(measurement['latency_std'])
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"Out of memory at size {size} bytes, stopping benchmark for {op}")
                        break
                    else:
                        print(f"Error at size {size} bytes for {op}: {e}")
                        continue
        
        # Create benchmark result
        parameters = {
            'operations': operations,
            'sizes_bytes': sizes_bytes,
            'runs': self.runs,
            'n_warmup': self.n_warmup
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
        operations: Optional[List[str]] = None
    ) -> None:
        """
        Plot the benchmark results.
        
        Args:
            result: BenchmarkResult to plot
            output_file: Output file path (default: auto-generated)
            operations: List of operations to include in the plot (default: all)
        """
        if output_file is None:
            device_name = sanitize_filename(result.device_info.get('name', 'unknown'))
            output_file = os.path.join(self.output_dir, f"{self.name}_{device_name}.png")
        
        # Filter operations if specified
        results = result.results
        if operations:
            results = {op: data for op, data in results.items() if op in operations}
        
        # Create plot title
        device_name = result.device_info.get('name', 'Unknown GPU')
        title = f"GPU Memory Latency ({device_name})"
        
        # Create plot
        plot_bandwidth_latency(
            results=results,
            title=title,
            output_file=output_file,
            include_bandwidth=False,
            include_latency=True
        )
        
        if self.verbose:
            print(f"Plot saved to {output_file}")


def main(
    device_id: int = 0,
    operations: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    runs: int = 10,
    n_warmup: int = 5,
    max_memory_fraction: float = 0.45,
    verbose: bool = True,
    output_file: Optional[str] = None
) -> BenchmarkResult:
    """
    Run a memory latency benchmark and plot the results.
    
    Args:
        device_id: CUDA device ID to use
        operations: List of memory operations to benchmark (default: ['zero', 'fill', 'copy'])
        output_dir: Directory to save results
        runs: Number of runs per size
        n_warmup: Number of warmup runs
        max_memory_fraction: Maximum fraction of free memory to use
        verbose: Whether to print progress information
        output_file: Output file path for the plot (default: auto-generated)
        
    Returns:
        BenchmarkResult containing the benchmark results
    """
    # Create benchmark
    benchmark = MemoryLatencyTest(
        device_id=device_id,
        operations=operations,
        output_dir=output_dir,
        runs=runs,
        n_warmup=n_warmup,
        max_memory_fraction=max_memory_fraction,
        verbose=verbose
    )
    
    # Run benchmark
    result = benchmark.run()
    
    # Plot results
    benchmark.plot(result, output_file=output_file)
    
    # Save results
    benchmark.save_results(result)
    
    return result
