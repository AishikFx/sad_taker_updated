"""
TRUE BATCH GFPGAN VRAM Utilization Test
Tests the new parallel processing implementation for maximum GPU utilization
"""

import torch
import sys
import os
import time
import psutil
import numpy as np
import cv2
from threading import Thread
import matplotlib.pyplot as plt

# Add SadTalker to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from src.utils.face_enhancer import TrueBatchGFPGANEnhancer, HighVRAMFaceEnhancer

class VRAMMonitor:
    """Monitor VRAM usage during processing"""
    
    def __init__(self):
        self.monitoring = False
        self.vram_usage = []
        self.timestamps = []
        self.peak_usage = 0
    
    def start_monitoring(self):
        """Start monitoring VRAM usage"""
        self.monitoring = True
        self.vram_usage = []
        self.timestamps = []
        self.peak_usage = 0
        
        def monitor():
            start_time = time.time()
            while self.monitoring:
                if torch.cuda.is_available():
                    current_usage = torch.cuda.memory_allocated() / 1e9  # GB
                    self.vram_usage.append(current_usage)
                    self.timestamps.append(time.time() - start_time)
                    self.peak_usage = max(self.peak_usage, current_usage)
                time.sleep(0.1)  # Sample every 100ms
        
        self.monitor_thread = Thread(target=monitor)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring and return results"""
        self.monitoring = False
        return {
            'timestamps': self.timestamps,
            'vram_usage': self.vram_usage,
            'peak_usage': self.peak_usage,
            'avg_usage': np.mean(self.vram_usage) if self.vram_usage else 0
        }

def create_test_images(num_images=100, size=(512, 512)):
    """Create realistic test images for face enhancement"""
    print(f"🎨 Creating {num_images} test images ({size[0]}x{size[1]})...")
    
    test_images = []
    for i in range(num_images):
        # Create a realistic face-like image
        img = np.random.randint(50, 200, (*size, 3), dtype=np.uint8)
        
        # Add face features
        center_x, center_y = size[0] // 2, size[1] // 2
        
        # Face oval
        cv2.ellipse(img, (center_x, center_y), (size[0]//3, size[1]//2), 0, 0, 360, (220, 180, 160), -1)
        
        # Eyes
        eye_y = center_y - size[1]//6
        cv2.circle(img, (center_x - size[0]//8, eye_y), size[0]//20, (50, 50, 50), -1)
        cv2.circle(img, (center_x + size[0]//8, eye_y), size[0]//20, (50, 50, 50), -1)
        
        # Nose
        cv2.circle(img, (center_x, center_y), size[0]//30, (200, 150, 120), -1)
        
        # Mouth
        mouth_y = center_y + size[1]//6
        cv2.ellipse(img, (center_x, mouth_y), (size[0]//12, size[1]//25), 0, 0, 180, (150, 100, 100), -1)
        
        test_images.append(img)
    
    return test_images

def test_vram_utilization():
    """Test VRAM utilization with different batch sizes"""
    
    print("🚀 TRUE BATCH GFPGAN VRAM UTILIZATION TEST")
    print("=" * 60)
    
    # Check GPU info
    if not torch.cuda.is_available():
        print("❌ No CUDA GPU detected. This test requires a CUDA-enabled GPU.")
        return
    
    gpu_name = torch.cuda.get_device_properties(0).name
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"🎯 GPU: {gpu_name}")
    print(f"🎯 Total VRAM: {gpu_memory_gb:.1f}GB")
    print(f"🎯 Target: Use 80-90% VRAM ({gpu_memory_gb * 0.8:.1f}-{gpu_memory_gb * 0.9:.1f}GB)")
    
    # Create test images
    test_images = create_test_images(num_images=50, size=(512, 512))
    
    # Test different batch sizes
    batch_sizes = [4, 8, 16, 24, 32, 48, 64]
    results = {}
    
    print("\n🧪 Testing TRUE BATCH processing with different batch sizes:")
    print("-" * 70)
    
    for batch_size in batch_sizes:
        print(f"\n🔬 Testing batch size: {batch_size}")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Initialize VRAM monitor
        monitor = VRAMMonitor()
        
        try:
            # Create true batch enhancer
            enhancer = TrueBatchGFPGANEnhancer(method='gfpgan')
            
            # Start monitoring
            monitor.start_monitoring()
            initial_vram = torch.cuda.memory_allocated() / 1e9
            
            # Time the enhancement
            start_time = time.time()
            
            # Process with true batch processing
            enhanced_images = enhancer.enhance_batch_parallel(test_images, batch_size=batch_size)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Stop monitoring
            monitor_results = monitor.stop_monitoring()
            
            # Calculate metrics
            fps = len(test_images) / processing_time
            vram_utilization = (monitor_results['peak_usage'] / gpu_memory_gb) * 100
            avg_vram_utilization = (monitor_results['avg_usage'] / gpu_memory_gb) * 100
            
            results[batch_size] = {
                'processing_time': processing_time,
                'fps': fps,
                'peak_vram_gb': monitor_results['peak_usage'],
                'avg_vram_gb': monitor_results['avg_usage'],
                'peak_vram_percent': vram_utilization,
                'avg_vram_percent': avg_vram_utilization,
                'success': True
            }
            
            # Color-coded output based on VRAM utilization
            if vram_utilization >= 80:
                status = "🟢 EXCELLENT"
            elif vram_utilization >= 60:
                status = "🟡 GOOD"
            elif vram_utilization >= 40:
                status = "🟠 MODERATE"
            else:
                status = "🔴 LOW"
            
            print(f"   ✅ Processing: {processing_time:.2f}s ({fps:.1f} FPS)")
            print(f"   📊 Peak VRAM: {monitor_results['peak_usage']:.2f}GB ({vram_utilization:.1f}%) {status}")
            print(f"   📈 Avg VRAM:  {monitor_results['avg_usage']:.2f}GB ({avg_vram_utilization:.1f}%)")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[batch_size] = {
                'success': False,
                'error': str(e)
            }
            monitor.stop_monitoring()
            torch.cuda.empty_cache()
    
    # Print summary and recommendations
    print("\n" + "=" * 70)
    print("📈 VRAM UTILIZATION ANALYSIS")
    print("=" * 70)
    
    successful_results = {k: v for k, v in results.items() if v.get('success', False)}
    
    if successful_results:
        # Find optimal batch size
        best_batch_size = max(successful_results.keys(), 
                            key=lambda x: successful_results[x]['peak_vram_percent'])
        best_result = successful_results[best_batch_size]
        
        print(f"\n🏆 OPTIMAL CONFIGURATION:")
        print(f"   Batch Size: {best_batch_size}")
        print(f"   VRAM Usage: {best_result['peak_vram_gb']:.2f}GB ({best_result['peak_vram_percent']:.1f}%)")
        print(f"   Performance: {best_result['fps']:.1f} FPS")
        
        # Check if we achieved target utilization
        if best_result['peak_vram_percent'] >= 80:
            print(f"\n🎉 SUCCESS! Achieved target VRAM utilization (80%+)")
            print(f"   Previous utilization was ~30% (4.4GB)")
            print(f"   New utilization is {best_result['peak_vram_percent']:.1f}% ({best_result['peak_vram_gb']:.2f}GB)")
            improvement = best_result['peak_vram_percent'] / 30
            print(f"   Improvement: {improvement:.1f}x better VRAM utilization!")
        else:
            print(f"\n⚠️  Target not reached. Peak utilization: {best_result['peak_vram_percent']:.1f}%")
            print("   Consider:")
            print("   • Larger batch sizes")
            print("   • Different image sizes")
            print("   • Memory optimization settings")
        
        # Performance comparison table
        print(f"\n📊 PERFORMANCE COMPARISON:")
        print(f"{'Batch Size':<12} {'VRAM Usage':<12} {'Utilization':<12} {'FPS':<8} {'Status'}")
        print("-" * 55)
        
        for batch_size in sorted(successful_results.keys()):
            result = successful_results[batch_size]
            vram_str = f"{result['peak_vram_gb']:.1f}GB"
            util_str = f"{result['peak_vram_percent']:.1f}%"
            fps_str = f"{result['fps']:.1f}"
            
            if result['peak_vram_percent'] >= 80:
                status = "🟢 Excellent"
            elif result['peak_vram_percent'] >= 60:
                status = "🟡 Good"
            else:
                status = "🔴 Low"
                
            print(f"{batch_size:<12} {vram_str:<12} {util_str:<12} {fps_str:<8} {status}")
    
    return results

def test_comparison_with_old_method():
    """Compare new method with old sequential processing"""
    print("\n⚡ PERFORMANCE COMPARISON: New vs Old Method")
    print("=" * 50)
    
    test_images = create_test_images(num_images=20, size=(512, 512))  # Smaller test for comparison
    
    # Test 1: Old method simulation (sequential processing)
    print("🐌 Testing OLD method (sequential processing)...")
    torch.cuda.empty_cache()
    
    old_monitor = VRAMMonitor()
    old_monitor.start_monitoring()
    
    start_time = time.time()
    
    # Simulate old sequential processing
    from src.utils.face_enhancer import OptimizedFaceEnhancer
    old_enhancer = OptimizedFaceEnhancer(method='gfpgan', optimization_level='medium')
    old_enhanced = old_enhancer.enhance_batch(test_images, batch_size=4)  # Small batch like before
    
    old_time = time.time() - start_time
    old_results = old_monitor.stop_monitoring()
    
    # Test 2: New method
    print("🚀 Testing NEW method (true batch processing)...")
    torch.cuda.empty_cache()
    
    new_monitor = VRAMMonitor()
    new_monitor.start_monitoring()
    
    start_time = time.time()
    
    new_enhancer = TrueBatchGFPGANEnhancer(method='gfpgan')
    new_enhanced = new_enhancer.enhance_batch_parallel(test_images, batch_size=32)
    
    new_time = time.time() - start_time
    new_results = new_monitor.stop_monitoring()
    
    # Compare results
    print("\n📊 COMPARISON RESULTS:")
    print("-" * 40)
    print(f"{'Metric':<20} {'Old Method':<15} {'New Method':<15} {'Improvement'}")
    print("-" * 65)
    
    old_fps = len(test_images) / old_time
    new_fps = len(test_images) / new_time
    fps_improvement = new_fps / old_fps
    
    old_vram_percent = (old_results['peak_usage'] / torch.cuda.get_device_properties(0).total_memory * 1e9) * 100
    new_vram_percent = (new_results['peak_usage'] / torch.cuda.get_device_properties(0).total_memory * 1e9) * 100
    vram_improvement = new_vram_percent / old_vram_percent
    
    print(f"{'Processing Time':<20} {old_time:.2f}s{'':<8} {new_time:.2f}s{'':<8} {new_time/old_time:.1f}x faster")
    print(f"{'FPS':<20} {old_fps:.1f}{'':<11} {new_fps:.1f}{'':<11} {fps_improvement:.1f}x")
    print(f"{'Peak VRAM':<20} {old_results['peak_usage']:.1f}GB{'':<8} {new_results['peak_usage']:.1f}GB{'':<8} {vram_improvement:.1f}x")
    print(f"{'VRAM Utilization':<20} {old_vram_percent:.1f}%{'':<9} {new_vram_percent:.1f}%{'':<9} {vram_improvement:.1f}x")
    
    return {
        'old': {'time': old_time, 'vram': old_results['peak_usage'], 'fps': old_fps},
        'new': {'time': new_time, 'vram': new_results['peak_usage'], 'fps': new_fps}
    }

if __name__ == "__main__":
    print("🎯 TESTING TRUE BATCH GFPGAN FOR MAXIMUM VRAM UTILIZATION")
    print("This test validates that we can achieve 80-90% VRAM usage instead of 30%")
    print()
    
    # Main VRAM utilization test
    vram_results = test_vram_utilization()
    
    # Performance comparison
    comparison_results = test_comparison_with_old_method()
    
    print("\n🎉 TEST COMPLETED!")
    print("\nNext steps:")
    print("1. Run your SadTalker with the optimized settings")
    print("2. Monitor VRAM usage - should see 80-90% instead of 30%")
    print("3. Enjoy much faster face enhancement processing!")
    
    # Final recommendations
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if gpu_memory_gb >= 14:
        recommended_batch = 48
    elif gpu_memory_gb >= 10:
        recommended_batch = 32
    else:
        recommended_batch = 24
    
    print(f"\n💡 RECOMMENDED SETTINGS FOR YOUR {gpu_memory_gb:.1f}GB GPU:")
    print(f"   --optimization_preset balanced")
    print(f"   Batch size will auto-adjust to ~{recommended_batch}")
    print(f"   Expected VRAM usage: 80-90% ({gpu_memory_gb * 0.8:.1f}-{gpu_memory_gb * 0.9:.1f}GB)")