#!/usr/bin/env python3
"""
Example script for running a memory bandwidth benchmark.

This script demonstrates how to use the benchmarking API to run
a memory bandwidth test and plot the results.
"""
import os
import sys
import torch

# Add the parent directory to the path so we can import the benchmarks package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.memory.bandwidth import MemoryBandwidthTest
from benchmarks.utils.device import get_device_info


def main():
    """Run a memory bandwidth benchmark."""
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return
    
    # Get the device ID (default to 0)
    device_id = 0
    
    # Print device information
    device_info = get_device_info(device_id)
    print(f"\nUsing GPU {device_id}: {device_info['name']}")
    print(f"CUDA capability: {device_info['compute_capability']}")
    print(f"Total memory: {device_info['total_memory'] / (1024**3):.2f} GB")
    
    # Create output directory if it doesn't exist
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create benchmark
    benchmark = MemoryBandwidthTest(
        device_id=device_id,
        operations=["zero", "fill", "copy"],  # Test these operations
        output_dir=output_dir,
        runs=10,                             # Number of runs per size
        n_warmup=5,                          # Number of warmup runs
        max_memory_fraction=0.45,            # Maximum fraction of free memory to use
        verbose=True                         # Print progress information
    )
    
    # Run benchmark
    print("\nRunning memory bandwidth benchmark...")
    result = benchmark.run()
    
    # Plot results
    print("\nPlotting results...")
    output_file = os.path.join(output_dir, f"memory_bandwidth_{device_info['name'].replace(' ', '_')}.png")
    benchmark.plot(
        result,
        output_file=output_file,
        plot_latency=True,
        plot_bandwidth=True
    )
    
    # Save results
    print("\nSaving results...")
    json_file = benchmark.save_results(result)
    
    print(f"\nBenchmark complete!")
    print(f"Results saved to {json_file}")
    print(f"Plot saved to {output_file}")


if __name__ == "__main__":
    main()
