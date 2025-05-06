import torch
import torch.distributed as dist
import argparse
import os
import numpy as np
from functools import partial
from profile_utils import profile_op
from plot_utils import plot_latency_results

def nccl_collective_op(op_name, tensor, group=None):
    if op_name == 'all_reduce':
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    elif op_name == 'all_gather':
        out = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(out, tensor, group=group)
    elif op_name == 'broadcast':
        dist.broadcast(tensor, src=0, group=group)
    elif op_name == 'reduce':
        dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM, group=group)
    elif op_name == 'reduce_scatter':
        input_tensor = torch.empty(
            (dist.get_world_size(), *tensor.shape), dtype=tensor.dtype, device=tensor.device)
        dist.reduce_scatter(tensor, list(input_tensor), op=dist.ReduceOp.SUM, group=group)
    else:
        raise ValueError(f"Unknown NCCL op: {op_name}")

def sweep_nccl_latency(
    op_names,
    sizes_bytes,
    device,
    runs=10,
    n_warmup=5,
    verbose=True
):
    results = {op: {'size_bytes': [], 'lat_ms_mean': [], 'lat_ms_std': [], 'bw_gbps': []} for op in op_names}
    group = dist.group.WORLD
    world_size = dist.get_world_size()
    for op in op_names:
        for size in sizes_bytes:
            tensor = torch.ones(size // 4, dtype=torch.float32, device=device)  # 4 bytes per float32
            if op == 'all_gather':
                # all_gather output is world_size * size
                def op_fn():
                    out = [torch.empty_like(tensor) for _ in range(world_size)]
                    dist.all_gather(out, tensor, group=group)
                bytes_xfer = size * world_size
            elif op == 'reduce_scatter':
                def op_fn():
                    input_tensor = torch.ones((world_size, tensor.numel()), dtype=tensor.dtype, device=device)
                    out_tensor = torch.empty_like(tensor)
                    dist.reduce_scatter(out_tensor, list(input_tensor), op=dist.ReduceOp.SUM, group=group)
                bytes_xfer = size
            else:
                def op_fn():
                    nccl_collective_op(op, tensor, group=group)
                bytes_xfer = size
            stats = profile_op(op_fn=op_fn, runs=runs, n_warmup=n_warmup)
            lat_ms = stats['mean'] * 1e3
            lat_std = stats['std'] * 1e3
            bw_gbps = (bytes_xfer * 8 / 1e9) / stats['mean'] if stats['mean'] > 0 else 0
            results[op]['size_bytes'].append(size)
            results[op]['lat_ms_mean'].append(lat_ms)
            results[op]['lat_ms_std'].append(lat_std)
            results[op]['bw_gbps'].append(bw_gbps)
            if verbose and dist.get_rank() == 0:
                print(f"{op} size={size} bytes: {lat_ms:.3f} ms, {bw_gbps:.2f} Gbps")
    return results

def plot_nccl_results(results, output_file=None, gpu_name=None):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    fig, ax1 = plt.subplots(figsize=(10, 6))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    for idx, (op, res) in enumerate(results.items()):
        sizes = np.array(res['size_bytes'])
        means = np.array(res['lat_ms_mean'])
        stds = np.array(res['lat_ms_std'])
        ax1.errorbar(sizes, means, yerr=stds, label=f'{op} Latency', fmt='-o', color=colors[idx % len(colors)])
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Tensor Size (bytes)')
    ax1.set_ylabel('Latency (ms)')
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: plot_utils.format_bytes(x, pos)))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: plot_utils.format_time(y/1e3, pos)))
    ax1.grid(True, which='both', axis='both', ls='--', lw=0.5)
    ax1.legend()
    plt.title(f'NCCL Collective Latency - {gpu_name}')
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="NCCL Collective Benchmark")
    parser.add_argument('--ops', type=str, nargs='+', default=['all_reduce', 'all_gather', 'broadcast'], help='NCCL ops to benchmark')
    parser.add_argument('--runs', type=int, default=10, help='Runs per size')
    parser.add_argument('--outfile', type=str, default=None, help='Output file for plot')
    parser.add_argument('--max-mem-frac', type=float, default=0.45, help='Fraction of free memory to use')
    args = parser.parse_args()

    import os
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend='nccl')
    free_mem_bytes = torch.cuda.mem_get_info(local_rank)[0]
    max_mem_bytes = int(free_mem_bytes * args.max_mem_frac)
    sizes_bytes = []
    i = 0
    while 2**i <= max_mem_bytes:
        sizes_bytes.append(2**i)
        i += 1
    results = sweep_nccl_latency(args.ops, sizes_bytes, device, runs=args.runs)
    gpu_name = torch.cuda.get_device_name(local_rank)
    outfile = args.outfile or f"nccl-bench-{gpu_name}.png"
    if dist.get_rank() == 0:
        plot_nccl_results(results, output_file=outfile, gpu_name=gpu_name)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
