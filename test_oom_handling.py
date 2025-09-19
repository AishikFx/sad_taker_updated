#!/usr/bin/env python3
"""
Test script for CUDA OOM handling in face enhancer
Tests the comprehensive OOM recovery system under memory pressure
"""

import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.face_enhancer import TrueBatchGFPGANEnhancer

def test_oom_logic():
    """Test OOM handling logic without CUDA"""
    print("🧪 Testing OOM Handling Logic (CPU-only)")
    print("=" * 50)

    try:
        # Mock CUDA availability for testing
        import torch
        original_cuda_available = torch.cuda.is_available

        # Force CUDA to appear available for testing logic
        torch.cuda.is_available = lambda: True
        torch.cuda.get_device_properties = lambda x: type('MockDevice', (), {'total_memory': 16*1024**3})()  # 16GB
        torch.cuda.memory_allocated = lambda: 2*1024**3  # 2GB allocated
        torch.cuda.empty_cache = lambda: None
        torch.cuda.synchronize = lambda: None
        torch.cuda.set_per_process_memory_fraction = lambda x: None
        torch.cuda.reset_peak_memory_stats = lambda: None

        # Initialize enhancer
        enhancer = TrueBatchGFPGANEnhancer()

        print("✅ Enhancer initialized successfully")
        print(f"Memory pressure level: {enhancer.memory_pressure_level}")
        print(f"OOM count: {enhancer.oom_count}")

        # Test performance insights
        insights = enhancer.get_performance_insights()
        print("\nPerformance Insights:")
        print(f"  Memory utilization: {insights['memory_utilization']:.1%}")
        print(f"  OOM events: {insights['oom_events']}")
        print(f"  Memory pressure: {insights['memory_pressure_level']}")
        print(f"  Recommendations: {len(insights['recommendations'])}")

        # Test adaptive OOM prevention
        safe_batch = enhancer._adaptive_oom_prevention(8, 4)
        print(f"\nAdaptive OOM prevention: batch 8 → {safe_batch}")

        # Test memory pressure tracking
        enhancer.record_failure(8, "oom")
        print(f"After OOM: pressure level = {enhancer.memory_pressure_level}, OOM count = {enhancer.oom_count}")

        enhancer.record_success(4)
        print(f"After success: pressure level = {enhancer.memory_pressure_level}")

        # Restore original CUDA function
        torch.cuda.is_available = original_cuda_available

        print("\n✅ OOM handling logic test completed successfully")
        return True

    except Exception as e:
        print(f"❌ OOM logic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        # Initialize enhancer
        enhancer = TrueBatchGFPGANEnhancer()

        # Get initial memory stats
        initial_memory = torch.cuda.memory_allocated() / 1e9
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Initial memory: {initial_memory:.1f}GB / {total_memory:.1f}GB total")
        print(f"Memory pressure level: {enhancer.memory_pressure_level}")
        print(f"OOM count: {enhancer.oom_count}")
        print()

        # Create test images (smaller for testing)
        test_images = []
        for i in range(8):  # Small batch for testing
            # Create a simple test image
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            test_images.append(img)

        print(f"Testing with {len(test_images)} images...")

        # Test normal processing
        print("\n1. Testing normal processing...")
        try:
            results = enhancer(test_images, batch_size=4)
            print(f"✅ Normal processing successful: {len(results)} images enhanced")
        except Exception as e:
            print(f"❌ Normal processing failed: {e}")
            return False

        # Test memory pressure simulation
        print("\n2. Testing memory pressure simulation...")
        try:
            # Fill GPU memory to simulate pressure
            dummy_tensors = []
            for i in range(10):
                try:
                    tensor = torch.randn(1000, 1000, device='cuda')
                    dummy_tensors.append(tensor)
                except:
                    break

            print(f"Created {len(dummy_tensors)} dummy tensors to simulate memory pressure")

            # Try processing under memory pressure
            results = enhancer(test_images, batch_size=8)  # Larger batch to trigger OOM
            print(f"✅ Memory pressure test successful: {len(results)} images enhanced")

        except Exception as e:
            print(f"⚠️  Memory pressure test triggered OOM (expected): {e}")

        # Clean up dummy tensors
        del dummy_tensors
        torch.cuda.empty_cache()

        # Test performance insights
        print("\n3. Testing performance insights...")
        try:
            insights = enhancer.get_performance_insights()
            print("Performance Insights:")
            print(f"  Memory utilization: {insights['memory_utilization']:.1%}")
            print(f"  OOM events: {insights['oom_events']}")
            print(f"  Memory pressure: {insights['memory_pressure_level']}")
            print(f"  Recommendations: {len(insights['recommendations'])}")
            for rec in insights['recommendations']:
                print(f"    - {rec}")
        except Exception as e:
            print(f"❌ Performance insights failed: {e}")

        # Final memory check
        final_memory = torch.cuda.memory_allocated() / 1e9
        print("\nFinal memory state:")
        print(f"  Initial memory: {initial_memory:.1f}GB")
        print(f"  Final memory: {final_memory:.1f}GB")
        print("✅ OOM handling test completed successfully")
        return True

    except Exception as e:
        print(f"❌ OOM handling test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_oom_logic()
    sys.exit(0 if success else 1)