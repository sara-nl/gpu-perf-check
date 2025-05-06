#!/usr/bin/env python3
"""
Example script for running an NCCL bandwidth benchmark.

This script demonstrates how to use the benchmarking API to run
an NCCL bandwidth test and plot the results.
"""
import os
import sys
import torch
import torch.distributed as dist

# Add the parent directory to the path so we can import the benchmarks package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.nccl.bandwidth import NCCLBandwidthTest
from benchmarks.utils.device import get_device_info, is_slurm_environment, get_slurm_info


def main():
    """Run an NCCL bandwidth benchmark."""
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return
    
    # Check if we're in a SLURM environment
    in_slurm = is_slurm_environment()
    if in_slurm:
        slurm_info = get_slurm_info()
        print(f"Running in SLURM environment:")
        for key, value in slurm_info.items():
            print(f"  {key}: {value}")
    
    # Create output directory if it doesn't exist
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create benchmark (this will initialize the distributed environment)
    benchmark = NCCLBandwidthTest(
        device_id=0,  # This will be overridden by LOCAL_RANK in a distributed setting
        operations=["all_reduce", "all_gather", "broadcast"],  # Test these operations
        output_dir=output_dir,
        runs=10,                             # Number of runs per size
        n_warmup=5,                          # Number of warmup runs
        max_memory_fraction=0.45,            # Maximum fraction of free memory to use
        verbose=True                         # Print progress information
    )
    
    try:
        # Get device information (only for rank 0)
        if dist.get_rank() == 0:
            device_id = torch.cuda.current_device()
            device_info = get_device_info(device_id)
            print(f"\nUsing GPU {device_id}: {device_info['name']}")
            print(f"CUDA capability: {device_info['compute_capability']}")
            print(f"Total memory: {device_info['total_memory'] / (1024**3):.2f} GB")
            print(f"World size: {dist.get_world_size()}")
        
        # Run benchmark
        if dist.get_rank() == 0:
            print("\nRunning NCCL bandwidth benchmark...")
        result = benchmark.run()
        
        # Only rank 0 plots and saves results
        if dist.get_rank() == 0:
            # Plot results
            print("\nPlotting results...")
            device_info = get_device_info(torch.cuda.current_device())
            output_file = os.path.join(output_dir, f"nccl_bandwidth_{device_info['name'].replace(' ', '_')}_n{dist.get_world_size()}.png")
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
    
    finally:
        # Clean up
        benchmark.cleanup()


if __name__ == "__main__":
    main()
