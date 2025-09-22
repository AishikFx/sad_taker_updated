"""
SadTalker Performance Optimization Configuration
Provides easy-to-use performance presets and detailed configuration options
"""

import torch

class OptimizationConfig:
    """Configuration manager for SadTalker optimizations"""
    
    # Performance presets for different use cases
    PRESETS = {
        "fast": {
            "optimization_level": "high",
            "face_renderer": {
                "batch_size": 12,
                "use_mixed_precision": False,  # Disabled for quality
                "aggressive_caching": False   # Disabled for quality
            },
            "seamless_clone": {
                "mode": "fast",  # Fast but quality-aware blending
                "use_parallel": True,
                "num_workers": 6
            },
            "face_enhancer": {
                "method": "standard",
                "batch_size": 12,
                "optimization_level": "high"
            },
            "description": "Good speed with maintained quality - 3x faster"
        },
        
        "fast": {
            "optimization_level": "high", 
            "face_renderer": {
                "batch_size": 12,
                "use_mixed_precision": False,  # Disabled for quality
                "aggressive_caching": False
            },
            "seamless_clone": {
                "mode": "fast",  # Feathered blending
                "use_parallel": True,
                "num_workers": 4
            },
            "face_enhancer": {
                "method": "gfpgan",
                "batch_size": 16,  # Increased for better VRAM utilization
                "optimization_level": "high",
                "use_high_vram_optimizer": True
            },
            "description": "Good speed with quality focus - 3x faster"
        },
        
        "balanced": {
            "optimization_level": "medium",
            "face_renderer": {
                "batch_size": 8,
                "use_mixed_precision": False,
                "aggressive_caching": False
            },
            "seamless_clone": {
                "mode": "balanced",  # Gaussian blending
                "use_parallel": True,
                "num_workers": 2
            },
            "face_enhancer": {
                "method": "gfpgan",
                "batch_size": 12,  # Increased for better VRAM utilization
                "optimization_level": "medium",
                "use_high_vram_optimizer": True
            },
            "description": "Balanced speed and quality - 3x faster"
        },
        
        "quality": {
            "optimization_level": "low",
            "face_renderer": {
                "batch_size": 4,
                "use_mixed_precision": False,
                "aggressive_caching": False
            },
            "seamless_clone": {
                "mode": "seamless",  # Original seamless clone
                "use_parallel": False,
                "num_workers": 1
            },
            "face_enhancer": {
                "method": "RestoreFormer",
                "batch_size": 8,  # Even quality mode can use higher batch sizes
                "optimization_level": "low",
                "use_high_vram_optimizer": True
            },
            "description": "Best quality, slower processing - original speed"
        }
    }
    
    def __init__(self, preset="balanced"):
        """Initialize with a preset configuration"""
        self.load_preset(preset)
        self.auto_detect_gpu_settings()
    
    def load_preset(self, preset_name):
        """Load a performance preset"""
        if preset_name not in self.PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(self.PRESETS.keys())}")
        
        self.preset_name = preset_name
        self.config = self.PRESETS[preset_name].copy()
        print(f"Loaded optimization preset: {preset_name}")
        print(f"Description: {self.config['description']}")
    
    def auto_detect_gpu_settings(self):
        """Auto-detect optimal settings based on GPU memory"""
        if not torch.cuda.is_available():
            print("No GPU detected, using CPU-optimized settings")
            self._apply_cpu_optimizations()
            return
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_name = torch.cuda.get_device_properties(0).name
        
        print(f"GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # Adjust batch sizes based on GPU memory
        if gpu_memory >= 24:
            scale_factor = 2.0
        elif gpu_memory >= 16:
            scale_factor = 1.5
        elif gpu_memory >= 12:
            scale_factor = 1.2
        elif gpu_memory >= 8:
            scale_factor = 1.0
        elif gpu_memory >= 6:
            scale_factor = 0.8
        else:
            scale_factor = 0.5
        
        # Scale batch sizes
        self.config["face_renderer"]["batch_size"] = max(1, int(
            self.config["face_renderer"]["batch_size"] * scale_factor))
        self.config["face_enhancer"]["batch_size"] = max(1, int(
            self.config["face_enhancer"]["batch_size"] * scale_factor))
        
        print(f"Adjusted batch sizes for {gpu_memory:.1f}GB GPU (scale: {scale_factor:.1f}x)")
        print(f"Face Renderer batch size: {self.config['face_renderer']['batch_size']}")
        print(f"Face Enhancer batch size: {self.config['face_enhancer']['batch_size']}")
    
    def _apply_cpu_optimizations(self):
        """Apply CPU-specific optimizations"""
        self.config["face_renderer"]["batch_size"] = 1
        self.config["face_renderer"]["use_mixed_precision"] = False
        self.config["face_enhancer"]["batch_size"] = 1
        self.config["seamless_clone"]["num_workers"] = min(2, torch.get_num_threads())
    
    def get_optimization_level(self):
        """Get the overall optimization level"""
        return self.config["optimization_level"]
    
    def get_face_renderer_config(self):
        """Get Face Renderer optimization config"""
        return self.config["face_renderer"]
    
    def get_seamless_clone_config(self):
        """Get seamless clone optimization config"""
        return self.config["seamless_clone"]
    
    def get_face_enhancer_config(self):
        """Get Face Enhancer optimization config"""
        return self.config["face_enhancer"]
    
    def print_performance_estimate(self):
        """Print estimated performance improvements"""
        speedups = {
            "fast": "3-4x faster", 
            "balanced": "2-3x faster",
            "quality": "Original speed"
        }
        
        print(f"\n=== Performance Estimate ===")
        print(f"Preset: {self.preset_name}")
        print(f"Expected speedup: {speedups.get(self.preset_name, 'Unknown')}")
        print(f"Face Renderer: Batch processing ({self.config['face_renderer']['batch_size']} frames)")
        print(f"Seamless Clone: {self.config['seamless_clone']['mode']} mode")
        print(f"Face Enhancer: {self.config['face_enhancer']['method']} with batch size {self.config['face_enhancer']['batch_size']}")
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU Memory: {gpu_memory:.1f}GB")
            if gpu_memory < 6:
                print("Low GPU memory detected. Consider 'fast' preset for better performance.")
            elif gpu_memory >= 12:
                print("Excellent GPU memory. All presets should work well.")
        
        print("="*30)

    @classmethod
    def list_presets(cls):
        """List all available presets with descriptions"""
        print("Available optimization presets:")
        for name, config in cls.PRESETS.items():
            print(f"  {name}: {config['description']}")

    def save_config(self, filepath):
        """Save current configuration to a file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"Configuration saved to {filepath}")

    def load_config(self, filepath):
        """Load configuration from a file"""
        import json
        with open(filepath, 'r') as f:
            self.config = json.load(f)
        print(f"Configuration loaded from {filepath}")


# Quick preset functions for convenience
def get_fast_config():
    """Get fast configuration (high speed, good quality)"""
    return OptimizationConfig("fast")

def get_balanced_config():
    """Get balanced configuration (default recommendation)"""
    return OptimizationConfig("balanced")

def get_quality_config():
    """Get quality configuration (best quality, original speed)"""
    return OptimizationConfig("quality")


# Performance measurement utilities
class PerformanceMonitor:
    """Monitor and report performance improvements"""
    
    def __init__(self):
        self.timings = {}
        self.start_times = {}
    
    def start_timing(self, operation):
        """Start timing an operation"""
        import time
        self.start_times[operation] = time.time()
    
    def end_timing(self, operation):
        """End timing an operation"""
        import time
        if operation in self.start_times:
            elapsed = time.time() - self.start_times[operation]
            self.timings[operation] = elapsed
            print(f"{operation}: {elapsed:.2f}s")
            return elapsed
        return None
    
    def get_total_time(self):
        """Get total processing time"""
        return sum(self.timings.values())
    
    def print_summary(self):
        """Print performance summary"""
        print("\n=== Performance Summary ===")
        for operation, time_taken in self.timings.items():
            print(f"{operation}: {time_taken:.2f}s")
        print(f"Total: {self.get_total_time():.2f}s")
        print("="*30)


# Example usage and testing
if __name__ == "__main__":
    # List available presets
    OptimizationConfig.list_presets()
    
    # Test different configurations
    configs = ["fast", "balanced", "quality"]
    
    for preset in configs:
        print(f"\n--- Testing {preset} preset ---")
        config = OptimizationConfig(preset)
        config.print_performance_estimate()