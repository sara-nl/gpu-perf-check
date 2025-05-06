import torch
import torch.nn as nn
import time
import gc
import math
from torch.amp import autocast, GradScaler

class BenchmarkNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BenchmarkNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        return self.model(x)

def estimate_max_tensor_size():
    """Estimate the maximum size of a tensor that will fit in GPU memory."""
    device = torch.device("cuda")
    total_memory = torch.cuda.get_device_properties(device).total_memory
    reserved_memory = torch.cuda.memory_reserved(device)
    allocated_memory = torch.cuda.memory_allocated(device)
    
    # Leave some buffer for other operations
    available_memory = total_memory - reserved_memory - allocated_memory
    buffer_factor = 0.9  # Use 80% of available memory
    available_memory *= buffer_factor
    
    # Each float32 value takes 4 bytes, but we'll account for mixed precision
    # Mixed precision will use float16 (2 bytes) for some operations
    bytes_per_element = 3  # Average between float32 and float16
    
    max_elements = int(available_memory / bytes_per_element)
    
    # Calculate dimensions for a square-ish tensor that fits
    dim_size = int(math.sqrt(max_elements))
    
    return dim_size

def benchmark():
    # Check for CUDA
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your GPU setup.")
        return
    
    device = torch.device("cuda")
    print(f"Using device: {torch.cuda.get_device_name(device)}")
    
    # Free up memory before starting
    torch.cuda.empty_cache()
    gc.collect()
    
    # Estimate dimensions to fill GPU memory
    dim_size = estimate_max_tensor_size()
    
    # Define neural network parameters
    input_size = dim_size
    hidden_size = 1024
    output_size = 256
    batch_size = 32
    
    # Create the model
    model = BenchmarkNN(input_size, hidden_size, output_size).to(device)
    
    # Compile the model
    print("Compiling model...")
    compiled_model = torch.compile(model)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(compiled_model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    scaler = GradScaler()
    
    # Generate random input that fills GPU memory
    print(f"Creating input tensor with shape: ({batch_size}, {input_size})")
    try:
        x = torch.randn(batch_size, input_size, device=device)
        y = torch.randn(batch_size, output_size, device=device)
    except RuntimeError as e:
        print(f"Error creating tensors: {e}")
        print("Reducing tensor size by half and trying again...")
        input_size = input_size // 2
        x = torch.randn(batch_size, input_size, device=device)
        y = torch.randn(batch_size, output_size, device=device)
    
    print(f"Input tensor shape: {x.shape}")
    
    # Warmup
    print("Performing warmup iterations...")
    for _ in range(10):
        with autocast(device_type='cuda', dtype=torch.float16):
            output = compiled_model(x)
            loss = loss_fn(output, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    
    # Benchmark
    num_iterations = 100
    print(f"\nRunning benchmark for {num_iterations} iterations...")
    
    # Track memory usage
    torch.cuda.reset_peak_memory_stats()
    start_memory = torch.cuda.memory_allocated(device)
    
    start_time = time.time()
    
    for i in range(num_iterations):
        iter_start = time.time()
        
        with autocast(device_type='cuda', dtype=torch.float16):
            output = compiled_model(x)
            loss = loss_fn(output, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        iter_end = time.time()
        
        if i % 10 == 0:
            print(f"Iteration {i}: {(iter_end - iter_start) * 1000:.2f} ms")
    
    end_time = time.time()
    
    # Calculate results
    peak_memory = torch.cuda.max_memory_allocated(device)
    total_time = end_time - start_time
    avg_time_per_iter = total_time / num_iterations
    
    # Report results
    print("\n===== Benchmark Results =====")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average time per iteration: {avg_time_per_iter * 1000:.4f} ms")
    print(f"Throughput: {num_iterations / total_time:.2f} iterations/second")
    print(f"Memory usage: {peak_memory / (1024**3):.2f} GB")
    print(f"Input tensor shape: {x.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print("============================")

if __name__ == "__main__":
    benchmark()