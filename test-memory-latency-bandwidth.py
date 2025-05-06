from latency_bandwidth_bench import sweep_memory_latency
from plot_utils import plot_latency_results
import torch
import argparse
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure GPU memory latency with PyTorch.")
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id (default: 0)')
    parser.add_argument('--size', type=int, default=1, help='Base tensor size in bytes (default: 1)')
    parser.add_argument('--plot', action='store_true', help='Plot the results')
    parser.add_argument('--outfile', type=str, default=None, help='Output file for plot (optional, default: autodetect from GPU name)')
    parser.add_argument('--runs', type=int, default=16, help='Number of runs per size (default: 5)')
    args = parser.parse_args()

    def sanitize_filename(name):
        return re.sub(r'[^\w\-]+', '-', name)

    # Get GPU total memory for the selected GPU
    total_mem_bytes = torch.cuda.get_device_properties(args.gpu).total_memory
    free_mem_bytes = torch.cuda.mem_get_info(args.gpu)[0]

    # Set a safer upper bound (e.g., 40% of free memory) to avoid OOM
    max_mem_bytes = int(free_mem_bytes * 0.45)

    # Nice size steps: add lower-end sizes and powers of 2 up to 0% of available GPU memory
    sizes_bytes = []
    # Only include powers of 2 above 512 bytes

    i = 0
    while 2**i <= max_mem_bytes:
        sizes_bytes.append(2**i)
        i += 1

    results = sweep_memory_latency(gpu_id=args.gpu, sizes_bytes=sizes_bytes, runs=args.runs, verbose=True)

    outfile = args.outfile
    gpu_name = torch.cuda.get_device_name(args.gpu)
    if outfile is None:
        outfile = sanitize_filename(gpu_name) + ".png"
    else:
        gpu_name = torch.cuda.get_device_name(args.gpu)
    if args.plot or True:  # Always plot
        plot_latency_results(results, output_file=outfile, gpu_name=gpu_name)
