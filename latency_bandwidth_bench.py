import torch
from functools import partial
from profile_utils import profile_op

def measure_memory_latency(gpu_id=0, tensor_size_bytes=1, runs=5, verbose=True, n_warmup=5):
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
            w_mean, w_std, r_mean, r_std = measure_memory_latency(gpu_id=gpu_id, tensor_size_bytes=size, runs=runs, verbose=verbose)
            results['size_bytes'].append(size)
            results['write_ms_mean'].append(w_mean * 1e3)
            results['write_ms_std'].append(w_std * 1e3)
            results['read_ms_mean'].append(r_mean * 1e3)
            results['read_ms_std'].append(r_std * 1e3)
            size_gib = size / 2**30
            write_bw = size_gib / w_mean if w_mean > 0 else 0
            read_bw = size_gib / r_mean if r_mean > 0 else 0
            results['write_bw_mean'].append(write_bw)
            results['write_bw_std'].append(0)
            results['read_bw_mean'].append(read_bw)
            results['read_bw_std'].append(0)
        except Exception as e:
            print(f"Error at size {size}: {e}")
            continue
    return results
