# Add this to the top of inference.py after imports
import torch
import os

# GPU Optimization Settings
def optimize_gpu_settings():
    """Configure optimal GPU settings for SadTalker"""
    if torch.cuda.is_available():
        # Enable memory pre-allocation
        torch.cuda.empty_cache()
        
        # Set memory growth to avoid fragmentation
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of available GPU memory
        
        # Enable optimized attention if available (for newer PyTorch versions)
        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except:
            pass
            
        # Optimize CUDNN settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = False
        
        print(f"GPU optimizations enabled. Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

# Call this function at the start of main()
# optimize_gpu_settings()