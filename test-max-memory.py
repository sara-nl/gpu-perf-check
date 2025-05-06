import torch
import time

def test_gpu_memory(gpu_id):
    """Test maximum allocatable memory on a specific GPU using binary search."""
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    print(f"Testing GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    
    # First phase: Find an upper bound that causes OOM
    size_mb = 2
    max_allocated_mb = 0
    allocated_tensors = []
    
    # Initial aggressive phase to find upper bound
    try:
        while True:
            # Convert MB to number of float32 elements (4 bytes per float)
            num_elements = int(size_mb * 1024 * 1024 / 4)
            
            print(f"Attempting to allocate {size_mb} MB on GPU {gpu_id}...")
            tensor = torch.zeros(num_elements, dtype=torch.float32, device=device)
            allocated_tensors.append(tensor)
            
            max_allocated_mb += size_mb
            
            # Gradually increase allocation size
            if size_mb < 1024:  # Less than 1GB
                size_mb *= 2
            else:
                size_mb += 1024  # Increase by 1GB when size is large
            
            time.sleep(0.1)
            
    except torch.cuda.OutOfMemoryError:
        print(f"First OOM on GPU {gpu_id} after allocating {max_allocated_mb} MB")
        # Free the last tensor that caused OOM
        print(f"except: {len(allocated_tensors)}")
        # if allocated_tensors:
        #     allocated_tensors.pop()
        #     max_allocated_mb -= size_mb
        
        # Second phase: Binary search for maximum memory
        print("Refining with binary search...")
        upper_bound = size_mb
        lower_bound = 1  # 1MB minimum precision
        
        while lower_bound > 1:
            mid_size = (upper_bound + lower_bound) // 2
            num_elements = int(mid_size * 1024 * 1024 / 4)
            
            print(f"Binary search: Trying {mid_size} MB on GPU {gpu_id}...")
            try:
                tensor = torch.zeros(num_elements, dtype=torch.float32, device=device)
                allocated_tensors.append(tensor)
                max_allocated_mb += mid_size
                lower_bound = mid_size + 1  # We can fit at least this much more
            except torch.cuda.OutOfMemoryError:
                upper_bound = mid_size  # This is too much
                torch.cuda.empty_cache()  # Clear the failed allocation
        
        print(f"Maximum allocatable memory on GPU {gpu_id}: {max_allocated_mb} MB")
        return max_allocated_mb
    except Exception as e:
        print(f"Error on GPU {gpu_id}: {e}")
        return max_allocated_mb

def main():
    """Test all available GPUs."""
    if not torch.cuda.is_available():
        print("No CUDA devices available")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} CUDA device(s)")
    
    results = {}
    
    for gpu_id in range(num_gpus):
        # Clear cache and reset device before testing
        torch.cuda.empty_cache()
        max_memory_mb = test_gpu_memory(gpu_id)
        results[gpu_id] = max_memory_mb
        # Give the system time to recover
        time.sleep(2)
    
    # Print summary
    print("\n===== RESULTS =====")
    for gpu_id, max_memory_mb in results.items():
        device_name = torch.cuda.get_device_name(gpu_id)
        total_memory_bytes = torch.cuda.get_device_properties(gpu_id).total_memory
        total_memory_mb = total_memory_bytes / (1024 * 1024)
        print(f"GPU {gpu_id} ({device_name}):")
        print(f"  - Total memory reported: {total_memory_mb:.2f} MB")
        print(f"  - Maximum allocatable memory: {max_memory_mb} MB")
        print(f"  - Utilization efficiency: {max_memory_mb/total_memory_mb*100:.2f}%")

if __name__ == "__main__":
    main()