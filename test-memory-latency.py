import torch
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def measure_memory_latency(gpu_id=0, tensor_size_bytes=1, runs=5, verbose=True, warmup=True):
    """
    Measures the latency of reading from and writing to GPU memory using a tensor of given size (in bytes).
    Repeats for `runs` times and returns (write_mean, write_std, read_mean, read_std) in seconds.
    Optionally performs a warmup iteration before timing.
    """
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    if verbose:
        print(f"Measuring memory latency on GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")

    num_elements = int(tensor_size_bytes)  # 1 byte per element

    # Warmup: run a write/read operation before actual timing
    if warmup:
        tensor = torch.zeros((num_elements,), dtype=torch.uint8, device=device)
        tensor.fill_(1)
        torch.cuda.synchronize()
        _ = torch.clone(tensor)
        torch.cuda.synchronize()

    write_latencies = []
    read_latencies = []
    for _ in range(runs):
        tensor = torch.zeros((num_elements,), dtype=torch.uint8, device=device)
        torch.cuda.synchronize()

        # Measure write latency
        start = time.perf_counter()
        tensor.fill_(1)
        torch.cuda.synchronize()
        write_latencies.append(time.perf_counter() - start)

        # Measure read latency (copy all elements)
        start = time.perf_counter()
        _ = torch.clone(tensor)
        torch.cuda.synchronize()
        read_latencies.append(time.perf_counter() - start)

    write_latencies = np.array(write_latencies)
    read_latencies = np.array(read_latencies)
    write_mean = write_latencies.mean()
    write_std = write_latencies.std()
    read_mean = read_latencies.mean()
    read_std = read_latencies.std()

    if verbose:
        print(f"Tensor size: {tensor_size_bytes} bytes")
        print(f"Write latency: {write_mean * 1e3:.3f} ± {write_std * 1e3:.3f} ms")
        print(f"Read latency: {read_mean * 1e3:.3f} ± {read_std * 1e3:.3f} ms")
    return write_mean, write_std, read_mean, read_std


def sweep_memory_latency(gpu_id=0, sizes_bytes=None, runs=5, verbose=True):
    """
    Sweep over a range of tensor sizes (in bytes), return dict with size, write/read avg/stdev latencies
    Warms up for the first few (smallest) sizes.
    """
    if sizes_bytes is None:
        sizes_bytes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    results = {'size_bytes': [], 'write_ms_mean': [], 'write_ms_std': [], 'read_ms_mean': [], 'read_ms_std': []}
    for idx, size in enumerate(sizes_bytes):
        try:
            # Warmup for the first 5 (smallest) sizes
            warmup = idx < 5
            w_mean, w_std, r_mean, r_std = measure_memory_latency(gpu_id=gpu_id, tensor_size_bytes=size, runs=runs, verbose=verbose, warmup=warmup)
            results['size_bytes'].append(size)
            results['write_ms_mean'].append(w_mean * 1e3)
            results['write_ms_std'].append(w_std * 1e3)
            results['read_ms_mean'].append(r_mean * 1e3)
            results['read_ms_std'].append(r_std * 1e3)
        except RuntimeError as e:
            if verbose:
                print(f"Failed for size {size} bytes: {e}")
            continue
    return results


def plot_latency_results(results, output_file=None, gpu_name=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.ticker as ticker

    sizes = np.array(results['size_bytes'])
    write_means = np.array(results['write_ms_mean'])
    write_stds = np.array(results['write_ms_std'])
    read_means = np.array(results['read_ms_mean'])
    read_stds = np.array(results['read_ms_std'])

    plt.figure(figsize=(10, 6))
    plt.errorbar(sizes, write_means, yerr=write_stds, label='Write', fmt='-o')
    plt.errorbar(sizes, read_means, yerr=read_stds, label='Read', fmt='-o')
    plt.xscale('log', base=2)
    plt.yscale('log')

    # Format x-ticks as human-readable bytes
    def format_bytes(x, pos=None):
        if x < 1024:
            return f"{int(x)} B"
        elif x < 1024**2:
            return f"{int(x/1024)} KiB"
        elif x < 1024**3:
            return f"{int(x/1024**2)} MiB"
        else:
            return f"{int(x/1024**3)} GiB"
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_bytes))
    ax.xaxis.set_major_locator(ticker.LogLocator(base=2))

    # Minor ticks/gridlines (log2, auto subs)
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=2, subs='auto', numticks=100))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.grid(True, which='major', axis='x', ls='-', lw=1, alpha=0.7)
    ax.grid(True, which='minor', axis='x', ls=':', lw=0.7, alpha=0.5)

    # Format y-ticks as human-readable time units (log scale)
    def format_time(y, pos=None):
        if y == 0:
            return "0"
        elif y < 1e-3:
            return f"{y*1e6:.0f} ns"
        elif y < 1:
            return f"{y*1e3:.1f} μs"
        elif y < 1e3:
            return f"{y:.2f} ms"
        else:
            return f"{y/1e3:.2f} s"
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_time))
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10))

    plt.xlabel('Tensor Size')
    plt.ylabel('Latency')
    title = 'GPU Memory Latency'
    if gpu_name:
        title += f' ({gpu_name})'
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", axis='y', ls="--", lw=0.5)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Measure GPU memory latency with PyTorch.")
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id (default: 0)')
    parser.add_argument('--size', type=int, default=1, help='Base tensor size in bytes (default: 1)')
    parser.add_argument('--plot', action='store_true', help='Plot the results')
    parser.add_argument('--outfile', type=str, default=None, help='Output file for plot (optional, default: autodetect from GPU name)')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs per size (default: 5)')
    args = parser.parse_args()

    def sanitize_filename(name):
        import re
        return re.sub(r'[^\w\-]+', '-', name)

    # Get GPU total memory for the selected GPU
    total_mem_bytes = torch.cuda.get_device_properties(args.gpu).total_memory
    free_mem_bytes = torch.cuda.mem_get_info(args.gpu)[0]

    # Set a safer upper bound (e.g., 40% of free memory) to avoid OOM
    max_mem_bytes = int(free_mem_bytes * 0.45)

    # Nice size steps: add lower-end sizes and powers of 2 up to 0% of available GPU memory
    sizes_bytes = []
    # Only include powers of 2 above 512 bytes

    i = 1
    while 2**i <= max_mem_bytes:
        sizes_bytes.append(2**i)
        i += 2

    results = sweep_memory_latency(gpu_id=args.gpu, sizes_bytes=sizes_bytes, runs=args.runs, verbose=True)

    outfile = args.outfile
    gpu_name = torch.cuda.get_device_name(args.gpu)
    if outfile is None:
        outfile = sanitize_filename(gpu_name) + ".png"
    else:
        gpu_name = torch.cuda.get_device_name(args.gpu)
    if args.plot or True:  # Always plot
        plot_latency_results(results, output_file=outfile, gpu_name=gpu_name)
