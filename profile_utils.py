import torch
import numpy as np

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
