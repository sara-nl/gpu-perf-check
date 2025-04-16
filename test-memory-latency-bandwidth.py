import torch
import numpy as np
from functools import partial



def profile_op(
    op_fn,
    setup_fn=None,
    n_warmup=5,
    runs=5,
):
    """
    Generic GPU profiling utility.
    - setup_fn: function to set up and return any resources/inputs (e.g. tensors)
    - op_fn: function that performs the operation to time (accepts setup_fn's outputs)
    - warmup_fn: function to warm up the op (accepts setup_fn's outputs)
    - teardown_fn: function to clean up (accepts setup_fn's outputs)
    Returns: dict with 'mean', 'std', 'latencies'
    """
    import torch
    import numpy as np
    
    for _ in range(n_warmup):
        if setup_fn:
            setup_fn()
        op_fn()
        torch.cuda.synchronize()
    
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
        latencies.append(start_evt.elapsed_time(end_evt) / 1e3)  # seconds

    latencies = np.array(latencies)
    return {
        'mean': float(latencies.mean()),
        'std': float(latencies.std()),
        'latencies': latencies.tolist(),
        'runs': runs
    }


def measure_memory_latency(gpu_id=0, tensor_size_bytes=1, runs=5, verbose=True, warmup=True):
    """
    Measures the latency of reading from and writing to GPU memory using a tensor of given size (in bytes).
    Uses profile_op for generic profiling.
    Returns (write_mean, write_std, read_mean, read_std) in seconds.
    """
    import torch
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    def setup():
        inp_tensor = torch.empty((tensor_size_bytes,), dtype=torch.uint8, device=device)
        outp_tensor = torch.empty((tensor_size_bytes,), dtype=torch.uint8, device=device)
        return inp_tensor, outp_tensor

    inp_tensor, outp_tensor = setup()

    def read_pre_setup(inp_tensor, outp_tensor):
        inp_tensor.zero_()
        outp_tensor.zero_()

    write_pre_setup = partial(inp_tensor.fill_, 255)
    write_op = inp_tensor.zero_

    read_pre_setup = partial(read_pre_setup, inp_tensor, outp_tensor)
    read_op = partial(outp_tensor.copy_, inp_tensor)

    write_stats = profile_op(op_fn=write_op, setup_fn=write_pre_setup, runs=runs)
    read_stats = profile_op(op_fn=read_op, setup_fn=read_pre_setup, runs=runs)

    if verbose:
        print(f"Tensor size: {tensor_size_bytes} bytes")
        print(f"Write latency: {write_stats['mean']*1e3:.3f} ± {write_stats['std']*1e3:.3f} ms")
        print(f"Read latency: {read_stats['mean']*1e3:.3f} ± {read_stats['std']*1e3:.3f} ms")
    return write_stats['mean'], write_stats['std'], read_stats['mean'], read_stats['std']


def sweep_memory_latency(gpu_id=0, sizes_bytes=None, runs=5, verbose=True):
    """
    Sweep over a range of tensor sizes (in bytes), return dict with size, write/read avg/stdev latencies and bandwidths
    Uses the refactored measure_memory_latency.
    """
    if sizes_bytes is None:
        sizes_bytes = [1]
    results = {
        'size_bytes': [],
        'write_ms_mean': [], 'write_ms_std': [],
        'read_ms_mean': [], 'read_ms_std': [],
        'write_bw_mean': [], 'write_bw_std': [],
        'read_bw_mean': [], 'read_bw_std': []
    }
    for _, size in enumerate(sizes_bytes):
        try:
            warmup = True
            w_mean, w_std, r_mean, r_std = measure_memory_latency(gpu_id=gpu_id, tensor_size_bytes=size, runs=runs, verbose=verbose, warmup=warmup)
            results['size_bytes'].append(size)
            results['write_ms_mean'].append(w_mean * 1e3)
            results['write_ms_std'].append(w_std * 1e3)
            results['read_ms_mean'].append(r_mean * 1e3)
            results['read_ms_std'].append(r_std * 1e3)

            # Bandwidth in GiB/s (latency in seconds)
            size_gib = size / 2**30
            write_bw = size_gib / w_mean if w_mean > 0 else 0
            read_bw = size_gib / r_mean if r_mean > 0 else 0
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
    ax1.set_yscale('log', base=10)

    # Format x-ticks as human-readable bytes
    def format_bytes(x, pos=None, suffix=""):
        if x < 1024:
            return f"{int(x)} B{suffix}"
        elif x < 1024**2:
            return f"{int(x/1024)} KiB{suffix}"
        elif x < 1024**3:
            return f"{int(x/1024**2)} MiB{suffix}"
        elif x < 1024**4:
            return f"{int(x/1024**3)} GiB{suffix}"
        else:
            return f"{int(x/1024**4)} TiB{suffix}"
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
    ax2.set_yscale('log', base=2)
    # Bandwidth is now in bytes/sec, so use the same formatter as bytes, with '/s' suffix
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: format_bytes(y, pos, suffix='/s')))
    ax2.set_ylabel('Bandwidth (bytes/sec)')

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


def benchmark_operation(
    operation_fn,
    op_args=None,
    op_kwargs=None,
    setup_fn=None,
    teardown_fn=None,
    runs=5,
    warmup=True
):
    """
    Generic benchmarking utility for GPU operations.
    - operation_fn: Callable to benchmark (should perform the operation to be timed)
    - op_args/op_kwargs: Arguments for the operation_fn
    - setup_fn/teardown_fn: Optional setup/cleanup callables (run before/after benchmarking)
    - runs: Number of timing runs
    - warmup: Whether to run a warmup iteration
    Returns: dict with timing statistics (mean, std, all latencies)
    """
    import torch
    import numpy as np

    op_args = op_args or ()
    op_kwargs = op_kwargs or {}

    if setup_fn:
        setup_fn()

    # Warmup
    if warmup:
        operation_fn(*op_args, **op_kwargs)
        torch.cuda.synchronize()

    latencies = []
    for _ in range(runs):
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        operation_fn(*op_args, **op_kwargs)
        end_evt.record()
        torch.cuda.synchronize()
        latencies.append(start_evt.elapsed_time(end_evt) / 1e3)  # seconds

    latencies = np.array(latencies)
    result = {
        "mean": float(latencies.mean()),
        "std": float(latencies.std()),
        "latencies": latencies.tolist(),
        "runs": runs
    }

    if teardown_fn:
        teardown_fn()

    return result


# Example: wrap memory copy as a benchmarkable operation

def memory_copy_op(inp_tensor, outp_tensor):
    outp_tensor.copy_(inp_tensor)


# Example usage for memory copy latency/throughput

def benchmark_memory_copy(gpu_id=0, tensor_size_bytes=1, runs=5, warmup=True):
    import torch
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    inp_tensor = torch.empty((tensor_size_bytes,), dtype=torch.uint8, device=device)
    outp_tensor = torch.empty((tensor_size_bytes,), dtype=torch.uint8, device=device)
    result = benchmark_operation(
        memory_copy_op,
        op_args=(inp_tensor, outp_tensor),
        runs=runs,
        warmup=warmup
    )
    # Optionally add bandwidth calculation
    result["bandwidth_GiBps"] = (tensor_size_bytes / result["mean"]) / (1024 ** 3) if result["mean"] > 0 else 0.0
    return result


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
