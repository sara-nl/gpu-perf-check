import torch

def get_gpu_memory_info():
    """
    Get memory information for all available GPUs using cudaMemGetInfo.
    Returns a list of tuples (free memory, total memory) in bytes for each GPU.
    """
    num_gpus = torch.cuda.device_count()
    memory_info = []
    
    for gpu_id in range(num_gpus):
        # Set current device
        torch.cuda.set_device(gpu_id)
        
        # Get memory info using cudaMemGetInfo
        free_memory, total_memory = torch.cuda.mem_get_info(gpu_id)
        
        memory_info.append((free_memory, total_memory))
    
    return memory_info

def main():
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return
        
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} CUDA device(s)")
    
    memory_info = get_gpu_memory_info()
    
    for gpu_id, (free_memory, total_memory) in enumerate(memory_info):
        device_name = torch.cuda.get_device_name(gpu_id)
        
        # Convert to MB for readability
        free_memory_mb = free_memory / (1024 * 1024)
        total_memory_mb = total_memory / (1024 * 1024)
        used_memory_mb = total_memory_mb - free_memory_mb
        
        print(f"GPU {gpu_id} ({device_name}):")
        print(f"  - Total memory: {total_memory_mb:.2f} MB")
        print(f"  - Free memory: {free_memory_mb:.2f} MB")
        print(f"  - Used memory: {used_memory_mb:.2f} MB")
        print(f"  - Memory utilization: {used_memory_mb/total_memory_mb*100:.2f}%")

if __name__ == "__main__":
    main()