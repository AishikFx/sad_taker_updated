# optimization_config.py
"""
Optimization configuration system for SadTalker
Provides presets for different speed/quality tradeoffs
"""

import torch
import psutil

class OptimizationConfig:
    """Configuration manager for SadTalker optimizations"""
    
    PRESETS = {
        "ultra_fast": {
            "face_enhancer": {
                "optimization_level": "extreme",
                "method": "lightweight",
                "batch_size": 16,
                "skip_bg_upsampler": True
            },
            "seamless_clone": {
                "use_fast_blend": True,
                "optimization_level": "extreme"
            },
            "face_renderer": {
                "batch_size": 8,
                "lightweight_mode": True
            },
            "description": "Fastest processing, basic enhancement only"
        },
        "fast": {
            "face_enhancer": {
                "optimization_level": "high", 
                "method": "gfpgan",
                "batch_size": 8,
                "skip_bg_upsampler": True
            },
            "seamless_clone": {
                "use_fast_blend": True,
                "optimization_level": "high"
            },
            "face_renderer": {
                "batch_size": 6,
                "lightweight_mode": False
            },
            "description": "Fast processing with GFPGAN, no background upsampling"
        },
        "balanced": {
            "face_enhancer": {
                "optimization_level": "medium",
                "method": "gfpgan", 
                "batch_size": 4,
                "skip_bg_upsampler": False
            },
            "seamless_clone": {
                "use_fast_blend": False,
                "optimization_level": "medium"
            },
            "face_renderer": {
                "batch_size": 4,
                "lightweight_mode": False
            },
            "description": "Balanced quality and speed"
        },
        "quality": {
            "face_enhancer": {
                "optimization_level": "low",
                "method": "RestoreFormer",
                "batch_size": 2,
                "skip_bg_upsampler": False
            },
            "seamless_clone": {
                "use_fast_blend": False,
                "optimization_level": "low"
            },
            "face_renderer": {
                "batch_size": 2,
                "lightweight_mode": False
            },
            "description": "Best quality, slower processing"
        }
    }
    
    def __init__(self, preset_name="balanced"):
        """Initialize with specified preset"""
        if preset_name not in self.PRESETS:
            print(f"Warning: Unknown preset '{preset_name}', using 'balanced'")
            preset_name = "balanced"
        
        self.preset_name = preset_name
        self.config = self.PRESETS[preset_name].copy()
        
        # Auto-adjust based on system capabilities
        self._auto_adjust_config()
        
        print(f"Using optimization preset: {preset_name}")
        print(f"Description: {self.config['description']}")
    
    def _auto_adjust_config(self):
        """Auto-adjust configuration based on system capabilities"""
        
        # GPU memory adjustments
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
            print(f"Detected GPU memory: {gpu_memory:.1f}GB")
            
            if gpu_memory < 4:
                # Low GPU memory - reduce batch sizes
                self.config["face_enhancer"]["batch_size"] = min(2, self.config["face_enhancer"]["batch_size"])
                self.config["face_renderer"]["batch_size"] = min(2, self.config["face_renderer"]["batch_size"])
                print("Reduced batch sizes for low GPU memory")
            elif gpu_memory >= 12:
                # High GPU memory - increase batch sizes for ultra_fast and fast
                if self.preset_name in ["ultra_fast", "fast"]:
                    self.config["face_enhancer"]["batch_size"] = min(32, self.config["face_enhancer"]["batch_size"] * 2)
                    self.config["face_renderer"]["batch_size"] = min(16, self.config["face_renderer"]["batch_size"] * 2)
                    print("Increased batch sizes for high GPU memory")
        else:
            # CPU only - use minimal batch sizes
            self.config["face_enhancer"]["batch_size"] = 1
            self.config["face_renderer"]["batch_size"] = 1
            self.config["face_enhancer"]["optimization_level"] = "extreme"
            print("CPU-only mode: using minimal batch sizes")
        
        # RAM adjustments
        ram_gb = psutil.virtual_memory().total / 1e9
        print(f"Detected RAM: {ram_gb:.1f}GB")
        
        if ram_gb < 8:
            # Low RAM - force extreme optimization
            self.config["face_enhancer"]["optimization_level"] = "extreme"
            self.config["seamless_clone"]["use_fast_blend"] = True
            print("Low RAM detected: forcing extreme optimizations")
    
    def get_face_enhancer_config(self):
        """Get face enhancer configuration"""
        return self.config["face_enhancer"]
    
    def get_seamless_clone_config(self):
        """Get seamless clone configuration"""
        return self.config["seamless_clone"]
    
    def get_face_renderer_config(self):
        """Get face renderer configuration"""
        return self.config["face_renderer"]
    
    def should_use_fast_blend(self):
        """Check if fast blending should be used"""
        return self.config["seamless_clone"]["use_fast_blend"]
    
    def get_optimization_level(self):
        """Get global optimization level"""
        return self.config["face_enhancer"]["optimization_level"]
    
    @classmethod
    def list_presets(cls):
        """List available presets with descriptions"""
        print("Available optimization presets:")
        for name, config in cls.PRESETS.items():
            print(f"  {name}: {config['description']}")
    
    @classmethod
    def get_recommended_preset(cls):
        """Get recommended preset based on system capabilities"""
        if not torch.cuda.is_available():
            return "ultra_fast"
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        ram_gb = psutil.virtual_memory().total / 1e9
        
        if gpu_memory >= 8 and ram_gb >= 16:
            return "balanced"
        elif gpu_memory >= 4 and ram_gb >= 8:
            return "fast"
        else:
            return "ultra_fast"


def detect_gpu_memory():
    """Detect available GPU memory"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_gb = gpu_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {gpu_memory_gb:.1f}GB")
        return gpu_memory_gb
    else:
        print("No GPU detected, using CPU")
        return 0


def benchmark_system():
    """Quick system benchmark for optimization recommendations"""
    print("=== System Benchmark ===")
    
    # GPU info
    gpu_memory = detect_gpu_memory()
    
    # RAM info
    ram_gb = psutil.virtual_memory().total / 1e9
    print(f"RAM: {ram_gb:.1f}GB")
    
    # CPU info
    cpu_count = psutil.cpu_count()
    print(f"CPU Cores: {cpu_count}")
    
    # Recommendations
    print("\n=== Recommendations ===")
    recommended = OptimizationConfig.get_recommended_preset()
    print(f"Recommended preset: {recommended}")
    
    OptimizationConfig.list_presets()
    
    return {
        "gpu_memory_gb": gpu_memory,
        "ram_gb": ram_gb,
        "cpu_cores": cpu_count,
        "recommended_preset": recommended
    }


class PerformanceMonitor:
    """Simple performance monitoring for profiling"""
    
    def __init__(self):
        self.start_time = None
        self.stages = {}
    
    def start(self):
        """Start monitoring"""
        import time
        self.start_time = time.time()
    
    def stage(self, stage_name):
        """Mark a stage"""
        import time
        if self.start_time:
            self.stages[stage_name] = time.time() - self.start_time
    
    def end(self):
        """End monitoring and return results"""
        import time
        if self.start_time:
            total_time = time.time() - self.start_time
            return {"total_time": total_time, "stages": self.stages}
        return {"total_time": 0, "stages": {}}


if __name__ == "__main__":
    # Run system benchmark
    benchmark_system()
    
    # Test all presets
    print("\n=== Testing Presets ===")
    for preset in OptimizationConfig.PRESETS.keys():
        print(f"\n--- {preset} ---")
        config = OptimizationConfig(preset)
        print(f"Face enhancer batch size: {config.get_face_enhancer_config()['batch_size']}")
        print(f"Use fast blend: {config.should_use_fast_blend()}")
        print(f"Optimization level: {config.get_optimization_level()}")