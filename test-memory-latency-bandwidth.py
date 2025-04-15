import torch
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def init_tensor(num_elements, device):
    return torch.randint(0, 256, (num_elements,), dtype=torch.uint8, device=device)

def measure_memory_latency(gpu_id=0, tensor_size_bytes=1, runs=5, verbose=True, warmup=True):
    """
    Measures the latency of reading from and writing to GPU memory using a tensor of given size (in bytes).
    Uses CUDA event timers for accurate timing.
    Pre-allocates the tensor to avoid allocation overhead.
    Returns (write_mean, write_std, read_mean, read_std) in seconds.
    """
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    if verbose:
        print(f"Measuring memory latency on GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")

    # Pre-allocate tensor
    inp_tensor = torch.empty((num_elements,), dtype=torch.uint8, device=device)
    outp_tensor = torch.empty((num_elements,), dtype=torch.uint8, device=device)

    # Warmup: run a write/read operation before actual timing
    if warmup:
        inp_tensor.fill_(1)
        torch.cuda.synchronize()
        outp_tensor.copy_(inp_tensor)
        torch.cuda.synchronize()

    write_latencies = []
    read_latencies = []
    for _ in range(runs):
        # Write timing
        # zeroes for as slow a write speed as possible
        inp_tensor.fill_(0)
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        inp_tensor.fill_(255)
        end_evt.record()
        torch.cuda.synchronize()
        write_latencies.append(start_evt.elapsed_time(end_evt) / 1e3)  # ms to s

        # Read timing (clone all elements)
        # zeroes for as quick a write speed as possible
        inp_tensor.fill_(0)
        outp_tensor.fill_(0)
        start_evt2 = torch.cuda.Event(enable_timing=True)
        end_evt2 = torch.cuda.Event(enable_timing=True)
        start_evt2.record()
        outp_tensor.copy_(inp_tensor)
        end_evt2.record()
        torch.cuda.synchronize()
        read_latencies.append(start_evt2.elapsed_time(end_evt2) / 1e3)  # ms to s

    write_latencies = np.array(write_latencies)
    read_latencies = np.array(read_latencies)

    write_mean = write_latencies.mean()
    read_mean = read_latencies.mean()

    read_std = read_latencies.std()
    write_std = write_latencies.std()

    if verbose:
        print(f"Tensor size: {tensor_size_bytes} bytes")
        print(f"Write latency: {write_mean * 1e3:.3f} ± {write_std * 1e3:.3f} ms")
        print(f"Read latency: {read_mean * 1e3:.3f} ± {read_std * 1e3:.3f} ms")
    return write_mean, write_std, read_mean, read_std


def sweep_memory_latency(gpu_id=0, sizes_bytes=None, runs=5, verbose=True):
    """
    Sweep over a range of tensor sizes (in bytes), return dict with size, write/read avg/stdev latencies and bandwidths
    Warms up for the first few (smallest) sizes.
    """
    if sizes_bytes is None:
        sizes_bytes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    results = {
        'size_bytes': [],
        'write_ms_mean': [], 'write_ms_std': [],
        'read_ms_mean': [], 'read_ms_std': [],
        'write_bw_mean': [], 'write_bw_std': [],
        'read_bw_mean': [], 'read_bw_std': []
    }
    for idx, size in enumerate(sizes_bytes):
        try:
            # Warmup for the first 5 (smallest) sizes
            warmup = True
            w_mean, w_std, r_mean, r_std = measure_memory_latency(gpu_id=gpu_id, tensor_size_bytes=size, runs=runs, verbose=verbose, warmup=warmup)
            results['size_bytes'].append(size)
            results['write_ms_mean'].append(w_mean * 1e3)
            results['write_ms_std'].append(w_std * 1e3)
            results['read_ms_mean'].append(r_mean * 1e3)
            results['read_ms_std'].append(r_std * 1e3)

            # Bandwidth in GiB/s
            size_gb = size / 2**30
            write_bw = size_gb / w_mean if w_mean > 0 else 0
            read_bw = size_gb / r_mean if r_mean > 0 else 0
            # For stddev, propagate errors (relative stddev)
            write_bw_std = write_bw * (w_std / w_mean) if w_mean > 0 else 0
            read_bw_std = read_bw * (r_std / r_mean) if r_mean > 0 else 0
            results['write_bw_mean'].append(write_bw)
            results['write_bw_std'].append(write_bw_std)
            results['read_bw_mean'].append(read_bw)
            results['read_bw_std'].append(read_bw_std)
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
    write_bw_means = np.array(results.get('write_bw_mean', []))
    write_bw_stds = np.array(results.get('write_bw_std', []))
    read_bw_means = np.array(results.get('read_bw_mean', []))
    read_bw_stds = np.array(results.get('read_bw_std', []))

    fig, ax1 = plt.subplots(figsize=(10, 6))
    l1 = ax1.errorbar(sizes, write_means, yerr=write_stds, label='Write Latency', fmt='-o', color='tab:blue')
    l2 = ax1.errorbar(sizes, read_means, yerr=read_stds, label='Read Latency', fmt='-o', color='tab:orange')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')

    # Format x-ticks as human-readable bytes
    def format_bytes(x, pos=None, suffix=""):
        if x < 1024:
            return f"{int(x)} B{suffix}"
        elif x < 1024**2:
            return f"{int(x/1024)} KiB{suffix}"
        elif x < 1024**3:
            return f"{int(x/1024**2)} MiB{suffix}"
        else:
            return f"{int(x/1024**3)} GiB{suffix}"
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: format_bytes(x, pos)))
    ax1.xaxis.set_major_locator(ticker.LogLocator(base=2))
    ax1.xaxis.set_minor_locator(ticker.LogLocator(base=2, subs='auto', numticks=100))
    ax1.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax1.grid(True, which='major', axis='x', ls='-', lw=1, alpha=0.7)
    ax1.grid(True, which='minor', axis='x', ls=':', lw=0.7, alpha=0.5)

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
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(format_time))
    ax1.yaxis.set_major_locator(ticker.LogLocator(base=10))

    ax1.set_xlabel('Tensor Size')
    ax1.set_ylabel('Latency')

    # Bandwidth on secondary axis
    ax2 = ax1.twinx()
    l3 = l4 = None
    if len(write_bw_means) > 0:
        l3 = ax2.errorbar(sizes, write_bw_means * 1024**3, yerr=write_bw_stds * 1024**3, label='Write Bandwidth', fmt='--s', color='tab:green')
    if len(read_bw_means) > 0:
        l4 = ax2.errorbar(sizes, read_bw_means * 1024**3, yerr=read_bw_stds * 1024**3, label='Read Bandwidth', fmt='--s', color='tab:red')
    ax2.set_yscale('log')
    # Bandwidth is now in bytes/sec, so use the same formatter as bytes, with '/s' suffix
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: format_bytes(y, pos, suffix='/s')))
    ax2.set_ylabel('Bandwidth (bytes/sec)')
    ax2.yaxis.label.set_color('tab:green')

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    if l3 is not None or l4 is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2
    ax1.legend(lines, labels, loc='upper left')

    title = 'GPU Memory Latency & Bandwidth'
    if gpu_name:
        title += f' ({gpu_name})'
    plt.title(title)
    ax1.grid(True, which="both", axis='y', ls="--", lw=0.5)
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
