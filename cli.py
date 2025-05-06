#!/usr/bin/env python3
"""
Command-line interface for GPU benchmarking.

This module provides a unified CLI for running memory and NCCL benchmarks.
"""
import argparse
import os
import sys
import torch
from typing import List, Optional, Dict, Any

from benchmarks.memory.bandwidth import MemoryBandwidthTest
from benchmarks.memory.latency import MemoryLatencyTest
from benchmarks.memory.operations import MEMORY_OPERATIONS
from benchmarks.nccl.bandwidth import NCCLBandwidthTest
from benchmarks.nccl.operations import NCCL_OPERATIONS
from benchmarks.utils.device import get_device_info, is_slurm_environment


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GPU Benchmarking Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Create subparsers for different benchmark types
    subparsers = parser.add_subparsers(dest="benchmark_type", help="Benchmark type")
    
    # Memory bandwidth benchmark
    mem_bw_parser = subparsers.add_parser(
        "memory-bandwidth",
        help="Benchmark GPU memory bandwidth"
    )
    _add_memory_args(mem_bw_parser)
    
    # Memory latency benchmark
    mem_lat_parser = subparsers.add_parser(
        "memory-latency",
        help="Benchmark GPU memory latency"
    )
    _add_memory_args(mem_lat_parser)
    
    # NCCL bandwidth benchmark
    nccl_parser = subparsers.add_parser(
        "nccl",
        help="Benchmark NCCL collective operations"
    )
    _add_nccl_args(nccl_parser)
    
    # Comparison benchmark
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare different operations or configurations"
    )
    _add_compare_args(compare_parser)
    
    # List available operations
    list_parser = subparsers.add_parser(
        "list-ops",
        help="List available operations"
    )
    list_parser.add_argument(
        "--type",
        choices=["memory", "nccl", "all"],
        default="all",
        help="Type of operations to list"
    )
    
    return parser.parse_args()


def _add_common_args(parser):
    """Add common arguments to a parser."""
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID to use"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file for plot (default: auto-generated)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of runs per size"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup runs"
    )
    parser.add_argument(
        "--max-memory-fraction",
        type=float,
        default=0.45,
        help="Maximum fraction of free memory to use"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=1,
        help="Minimum tensor size in bytes"
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Maximum tensor size in bytes (default: auto-calculated)"
    )


def _add_memory_args(parser):
    """Add memory benchmark specific arguments to a parser."""
    _add_common_args(parser)
    
    # Available memory operations
    available_ops = list(MEMORY_OPERATIONS.keys())
    
    parser.add_argument(
        "--operations",
        type=str,
        nargs="+",
        default=["zero", "fill", "copy"],
        choices=available_ops,
        help="Memory operations to benchmark"
    )
    parser.add_argument(
        "--plot-latency",
        action="store_true",
        default=True,
        help="Plot latency"
    )
    parser.add_argument(
        "--plot-bandwidth",
        action="store_true",
        default=True,
        help="Plot bandwidth"
    )


def _add_nccl_args(parser):
    """Add NCCL benchmark specific arguments to a parser."""
    _add_common_args(parser)
    
    # Available NCCL operations
    available_ops = list(NCCL_OPERATIONS.keys())
    
    parser.add_argument(
        "--operations",
        type=str,
        nargs="+",
        default=["all_reduce", "all_gather", "broadcast"],
        choices=available_ops,
        help="NCCL operations to benchmark"
    )
    parser.add_argument(
        "--plot-latency",
        action="store_true",
        default=True,
        help="Plot latency"
    )
    parser.add_argument(
        "--plot-bandwidth",
        action="store_true",
        default=True,
        help="Plot bandwidth"
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Running in a SLURM environment"
    )


def _add_compare_args(parser):
    """Add comparison benchmark specific arguments to a parser."""
    _add_common_args(parser)
    
    # All available operations
    memory_ops = list(MEMORY_OPERATIONS.keys())
    nccl_ops = list(NCCL_OPERATIONS.keys())
    
    parser.add_argument(
        "--benchmark-type",
        type=str,
        choices=["memory-bandwidth", "memory-latency", "nccl"],
        required=True,
        help="Type of benchmark to compare"
    )
    parser.add_argument(
        "--operations",
        type=str,
        nargs="+",
        required=True,
        help="Operations to compare"
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["bandwidth_mean", "latency_mean"],
        default="bandwidth_mean",
        help="Metric to compare"
    )


def run_memory_bandwidth(args):
    """Run memory bandwidth benchmark."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create benchmark
    benchmark = MemoryBandwidthTest(
        device_id=args.gpu,
        operations=args.operations,
        output_dir=args.output_dir,
        runs=args.runs,
        n_warmup=args.warmup,
        max_memory_fraction=args.max_memory_fraction,
        verbose=args.verbose
    )
    
    # Generate sizes
    sizes = benchmark.generate_sizes(
        min_bytes=args.min_size,
        max_bytes=args.max_size
    )
    
    # Run benchmark
    result = benchmark.run(sizes_bytes=sizes)
    
    # Plot results
    benchmark.plot(
        result,
        output_file=args.output_file,
        plot_latency=args.plot_latency,
        plot_bandwidth=args.plot_bandwidth
    )
    
    # Save results
    benchmark.save_results(result)
    
    return result


def run_memory_latency(args):
    """Run memory latency benchmark."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create benchmark
    benchmark = MemoryLatencyTest(
        device_id=args.gpu,
        operations=args.operations,
        output_dir=args.output_dir,
        runs=args.runs,
        n_warmup=args.warmup,
        max_memory_fraction=args.max_memory_fraction,
        verbose=args.verbose
    )
    
    # Generate sizes
    sizes = benchmark.generate_sizes(
        min_bytes=args.min_size,
        max_bytes=args.max_size
    )
    
    # Run benchmark
    result = benchmark.run(sizes_bytes=sizes)
    
    # Plot results
    benchmark.plot(
        result,
        output_file=args.output_file
    )
    
    # Save results
    benchmark.save_results(result)
    
    return result


def run_nccl_benchmark(args):
    """Run NCCL benchmark."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if we're in a SLURM environment
    if args.slurm or is_slurm_environment():
        print("Running in SLURM environment")
    
    # Create benchmark
    benchmark = NCCLBandwidthTest(
        device_id=args.gpu,
        operations=args.operations,
        output_dir=args.output_dir,
        runs=args.runs,
        n_warmup=args.warmup,
        max_memory_fraction=args.max_memory_fraction,
        verbose=args.verbose
    )
    
    # Generate sizes
    sizes = benchmark.generate_sizes(
        min_bytes=args.min_size,
        max_bytes=args.max_size
    )
    
    try:
        # Run benchmark
        result = benchmark.run(sizes_bytes=sizes)
        
        # Only rank 0 plots and saves results
        if torch.distributed.get_rank() == 0:
            # Plot results
            benchmark.plot(
                result,
                output_file=args.output_file,
                plot_latency=args.plot_latency,
                plot_bandwidth=args.plot_bandwidth
            )
            
            # Save results
            benchmark.save_results(result)
        
        return result
    finally:
        # Clean up
        benchmark.cleanup()


def run_comparison(args):
    """Run comparison benchmark."""
    print("Comparison benchmark not yet implemented")
    return None


def list_operations(args):
    """List available operations."""
    if args.type in ["memory", "all"]:
        print("\nAvailable memory operations:")
        for op, info in MEMORY_OPERATIONS.items():
            print(f"  - {op}: {info['description']}")
    
    if args.type in ["nccl", "all"]:
        print("\nAvailable NCCL operations:")
        for op, info in NCCL_OPERATIONS.items():
            print(f"  - {op}: {info['description']}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available")
        sys.exit(1)
    
    # Print device information
    if args.benchmark_type != "list-ops" and getattr(args, "verbose", False):
        device_id = getattr(args, "gpu", 0)
        device_info = get_device_info(device_id)
        print(f"\nUsing GPU {device_id}: {device_info['name']}")
        print(f"CUDA capability: {device_info['compute_capability']}")
        print(f"Total memory: {device_info['total_memory'] / (1024**3):.2f} GB")
    
    # Run the appropriate benchmark
    if args.benchmark_type == "memory-bandwidth":
        run_memory_bandwidth(args)
    elif args.benchmark_type == "memory-latency":
        run_memory_latency(args)
    elif args.benchmark_type == "nccl":
        run_nccl_benchmark(args)
    elif args.benchmark_type == "compare":
        run_comparison(args)
    elif args.benchmark_type == "list-ops":
        list_operations(args)
    else:
        print("Please specify a benchmark type")
        sys.exit(1)


if __name__ == "__main__":
    main()
