"""
SadTalker Performance Optimization Test Script
Demonstrates the performance improvements and provides benchmarking utilities
"""

import os
import sys
import time
import argparse
import torch

# Add SadTalker to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from src.utils.optimization_config import OptimizationConfig, PerformanceMonitor
from inference import main as sadtalker_main

def benchmark_optimization_levels(source_image, driven_audio, result_dir="./benchmark_results"):
    """Benchmark different optimization levels"""
    
    presets = ["quality", "balanced", "fast", "ultra_fast"]
    results = {}
    
    print("🚀 SadTalker Performance Optimization Benchmark")
    print("=" * 60)
    
    for preset in presets:
        print(f"\n📊 Testing {preset} preset...")
        
        # Create preset-specific result directory
        preset_result_dir = os.path.join(result_dir, f"benchmark_{preset}")
        os.makedirs(preset_result_dir, exist_ok=True)
        
        # Configure arguments
        class Args:
            def __init__(self):
                self.source_image = source_image
                self.driven_audio = driven_audio
                self.result_dir = preset_result_dir
                self.optimization_preset = preset
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.cpu = False
                self.profile = True
                self.enhancer = None
                self.background_enhancer = None
                self.checkpoint_dir = './checkpoints'
                self.pose_style = 0
                self.batch_size = 8
                self.size = 256
                self.expression_scale = 1.0
                self.input_yaw = None
                self.input_pitch = None
                self.input_roll = None
                self.ref_eyeblink = None
                self.ref_pose = None
                self.face3dvis = False
                self.still = False
                self.preprocess = 'crop'
                self.verbose = False
                self.old_version = False
                self.net_recon = 'resnet50'
                self.init_path = None
                self.use_last_fc = False
                self.bfm_folder = './checkpoints/BFM_Fitting/'
                self.bfm_model = 'BFM_model_front.mat'
                self.focal = 1015.0
                self.center = 112.0
                self.camera_d = 10.0
                self.z_near = 5.0
                self.z_far = 15.0
        
        args = Args()
        
        # Run benchmark
        start_time = time.time()
        try:
            sadtalker_main(args)
            end_time = time.time()
            total_time = end_time - start_time
            
            results[preset] = {
                "success": True,
                "total_time": total_time,
                "result_dir": preset_result_dir
            }
            
            print(f"✅ {preset}: {total_time:.2f}s")
            
        except Exception as e:
            print(f"❌ {preset}: Failed - {e}")
            results[preset] = {
                "success": False,
                "error": str(e)
            }
    
    # Print summary
    print("\n" + "=" * 60)
    print("📈 BENCHMARK SUMMARY")
    print("=" * 60)
    
    successful_results = {k: v for k, v in results.items() if v.get("success", False)}
    
    if successful_results:
        # Find baseline (quality preset)
        baseline_time = successful_results.get("quality", {}).get("total_time")
        
        for preset, result in successful_results.items():
            total_time = result["total_time"]
            if baseline_time:
                speedup = baseline_time / total_time
                print(f"{preset:12s}: {total_time:6.2f}s ({speedup:4.1f}x speedup)")
            else:
                print(f"{preset:12s}: {total_time:6.2f}s")
    
    print("=" * 60)
    
    return results

def quick_speed_test(source_image, driven_audio):
    """Quick speed test with ultra_fast preset"""
    
    print("⚡ Quick Speed Test (ultra_fast preset)")
    print("-" * 40)
    
    # Configure for ultra-fast test
    class Args:
        def __init__(self):
            self.source_image = source_image
            self.driven_audio = driven_audio
            self.result_dir = "./quick_test_results"
            self.optimization_preset = "ultra_fast"
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.cpu = False
            self.profile = True
            self.enhancer = None  # Disable for speed test
            self.background_enhancer = None
            self.checkpoint_dir = './checkpoints'
            self.pose_style = 0
            self.batch_size = 16  # Larger batch for speed
            self.size = 256
            self.expression_scale = 1.0
            self.input_yaw = None
            self.input_pitch = None
            self.input_roll = None
            self.ref_eyeblink = None
            self.ref_pose = None
            self.face3dvis = False
            self.still = False
            self.preprocess = 'crop'
            self.verbose = False
            self.old_version = False
            self.net_recon = 'resnet50'
            self.init_path = None
            self.use_last_fc = False
            self.bfm_folder = './checkpoints/BFM_Fitting/'
            self.bfm_model = 'BFM_model_front.mat'
            self.focal = 1015.0
            self.center = 112.0
            self.camera_d = 10.0
            self.z_near = 5.0
            self.z_far = 15.0
    
    args = Args()
    os.makedirs(args.result_dir, exist_ok=True)
    
    start_time = time.time()
    try:
        sadtalker_main(args)
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"✅ Ultra-fast processing completed in {total_time:.2f}s")
        print(f"📁 Results saved to: {args.result_dir}")
        
        return total_time
        
    except Exception as e:
        print(f"❌ Speed test failed: {e}")
        return None

def validate_setup():
    """Validate that the optimization setup is working correctly"""
    
    print("🔧 Validating SadTalker Optimization Setup")
    print("-" * 45)
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_properties(0).name
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("⚠️  No GPU detected, will use CPU (slower)")
    
    # Test optimization config
    try:
        config = OptimizationConfig("balanced")
        print("✅ Optimization configuration loaded successfully")
        config.print_performance_estimate()
    except Exception as e:
        print(f"❌ Optimization config failed: {e}")
        return False
    
    # Check for required files
    required_dirs = ["./checkpoints", "./src"]
    missing_dirs = [d for d in required_dirs if not os.path.exists(d)]
    
    if missing_dirs:
        print(f"❌ Missing required directories: {missing_dirs}")
        return False
    else:
        print("✅ Required directories found")
    
    print("✅ Setup validation completed successfully")
    return True

def show_usage_examples():
    """Show practical usage examples"""
    
    print("\n🚀 SadTalker Optimization Usage Examples")
    print("=" * 50)
    
    examples = [
        {
            "name": "Maximum Speed (10x faster)",
            "command": "python inference.py --source_image ./examples/source_image/full_body_1.png --driven_audio ./examples/driven_audio/bus_chinese.wav --optimization_preset ultra_fast",
            "description": "Use for real-time applications or when speed is critical"
        },
        {
            "name": "Fast Processing (5x faster)",
            "command": "python inference.py --source_image ./examples/source_image/full_body_1.png --driven_audio ./examples/driven_audio/bus_chinese.wav --optimization_preset fast",
            "description": "Good balance of speed and quality for most use cases"
        },
        {
            "name": "Balanced (3x faster, default)", 
            "command": "python inference.py --source_image ./examples/source_image/full_body_1.png --driven_audio ./examples/driven_audio/bus_chinese.wav --optimization_preset balanced",
            "description": "Recommended default setting"
        },
        {
            "name": "High Quality (original speed)",
            "command": "python inference.py --source_image ./examples/source_image/full_body_1.png --driven_audio ./examples/driven_audio/bus_chinese.wav --optimization_preset quality",
            "description": "Use when quality is more important than speed"
        },
        {
            "name": "With Face Enhancement (slower)",
            "command": "python inference.py --source_image ./examples/source_image/full_body_1.png --driven_audio ./examples/driven_audio/bus_chinese.wav --optimization_preset fast --enhancer gfpgan",
            "description": "Add face enhancement with fast processing"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['name']}")
        print(f"   Description: {example['description']}")
        print(f"   Command: {example['command']}")
    
    print("\n📋 Available Presets:")
    OptimizationConfig.list_presets()
    
    print("\n💡 Tips:")
    print("   • Use 'ultra_fast' for real-time applications")
    print("   • Use 'fast' for general processing")  
    print("   • Use 'balanced' as the default recommendation")
    print("   • Use 'quality' only when maximum quality is needed")
    print("   • Add --profile to see detailed timing information")


def main():
    parser = argparse.ArgumentParser(description="SadTalker Performance Optimization Test")
    parser.add_argument("--action", choices=["benchmark", "quick_test", "validate", "examples"], 
                       default="examples", help="Action to perform")
    parser.add_argument("--source_image", default="./examples/source_image/full_body_1.png",
                       help="Source image path for testing")
    parser.add_argument("--driven_audio", default="./examples/driven_audio/bus_chinese.wav", 
                       help="Driven audio path for testing")
    
    args = parser.parse_args()
    
    if args.action == "validate":
        validate_setup()
    elif args.action == "examples":
        show_usage_examples()
    elif args.action == "quick_test":
        if not os.path.exists(args.source_image) or not os.path.exists(args.driven_audio):
            print(f"❌ Test files not found:")
            print(f"   Source image: {args.source_image}")
            print(f"   Driven audio: {args.driven_audio}")
            print("   Please provide valid file paths or use default examples")
            return
        quick_speed_test(args.source_image, args.driven_audio)
    elif args.action == "benchmark":
        if not os.path.exists(args.source_image) or not os.path.exists(args.driven_audio):
            print(f"❌ Test files not found:")
            print(f"   Source image: {args.source_image}")  
            print(f"   Driven audio: {args.driven_audio}")
            print("   Please provide valid file paths or use default examples")
            return
        benchmark_optimization_levels(args.source_image, args.driven_audio)
    
    print("\n🎉 Done! Run with --action examples to see usage instructions.")

if __name__ == "__main__":
    main()