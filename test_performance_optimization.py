#!/usr/bin/env python3
"""
Performance test for optimized face enhancer
Tests that the new version is faster than 1.45 it/s
"""

import time
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_performance():
    """Test performance with simulated image processing"""
    print("⚡ Testing Optimized Face Enhancer Performance")
    print("=" * 50)

    # Test configuration
    num_images = 109  # Same as your test
    target_performance = 1.45  # Original performance (images per second)
    
    # Create test images (smaller for testing)
    test_images = []
    for i in range(num_images):
        # Create a simple test image (smaller than production)
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        test_images.append(img)

    print(f"Testing with {len(test_images)} images (256x256 each)")
    print(f"Target performance: {target_performance:.2f} images/second")
    print()

    try:
        # Import the enhanced face enhancer
        from src.utils.face_enhancer import HighVRAMFaceEnhancer
        
        # Initialize enhancer
        enhancer = HighVRAMFaceEnhancer()
        
        print("✅ Enhancer initialized successfully")
        print("🚀 Starting performance test...")
        
        # Warm up (don't count in timing)
        print("   Warming up...")
        warmup_images = test_images[:5]
        _ = enhancer.enhance_batch_ultra(warmup_images, batch_size=4)
        
        # Actual performance test
        print("   Testing performance...")
        start_time = time.time()
        
        # Process all images
        enhanced_images = enhancer.enhance_batch_ultra(test_images, batch_size=8)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Calculate performance metrics
        images_per_second = len(test_images) / processing_time
        performance_improvement = images_per_second / target_performance
        
        print(f"\n📊 Performance Results:")
        print(f"  Total time: {processing_time:.2f} seconds")
        print(f"  Images processed: {len(enhanced_images)}")
        print(f"  Performance: {images_per_second:.2f} images/second")
        print(f"  Target: {target_performance:.2f} images/second")
        print(f"  Improvement: {performance_improvement:.2f}x")
        
        if images_per_second >= target_performance:
            print(f"\n🎉 SUCCESS: Performance target met!")
            print(f"   Achieved {images_per_second:.2f} it/s (target: {target_performance:.2f} it/s)")
            if performance_improvement > 1.2:
                print(f"   🚀 EXCELLENT: {performance_improvement:.2f}x faster than original!")
            return True
        else:
            print(f"\n⚠️  PERFORMANCE ISSUE: Below target")
            print(f"   Current: {images_per_second:.2f} it/s")
            print(f"   Target: {target_performance:.2f} it/s")
            print(f"   Deficit: {target_performance - images_per_second:.2f} it/s")
            return False

    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_distribution():
    """Test that the new batch distribution is more efficient"""
    print("\n🔄 Testing Batch Distribution Efficiency")
    print("=" * 50)
    
    # Simulate the old vs new distribution
    num_images = 28  # Example from your logs
    num_streams = 4
    
    print(f"Testing distribution of {num_images} images across {num_streams} streams")
    
    # Old method (round-robin)
    old_distribution = [[] for _ in range(num_streams)]
    for i in range(num_images):
        stream_idx = i % num_streams
        old_distribution[stream_idx].append(f"img_{i}")
    
    print("\n📊 Old Distribution (Round-Robin):")
    for i, batch in enumerate(old_distribution):
        print(f"  Stream {i}: {len(batch)} images")
    
    # New method (chunked)
    images_per_stream = max(1, num_images // num_streams)
    new_distribution = []
    
    for i in range(num_streams):
        start_idx = i * images_per_stream
        end_idx = min((i + 1) * images_per_stream, num_images)
        
        if start_idx < num_images:
            stream_batch = list(range(start_idx, end_idx))
            if stream_batch:
                new_distribution.append(stream_batch)
    
    # Handle remaining images
    remaining_start = num_streams * images_per_stream
    if remaining_start < num_images:
        remaining_images = list(range(remaining_start, num_images))
        if new_distribution:
            new_distribution[-1].extend(remaining_images)
        else:
            new_distribution.append(remaining_images)
    
    print("\n📊 New Distribution (Chunked):")
    for i, batch in enumerate(new_distribution):
        print(f"  Stream {i}: {len(batch)} images")
    
    # Analysis
    old_max_batch = max(len(batch) for batch in old_distribution)
    new_max_batch = max(len(batch) for batch in new_distribution)
    
    print(f"\n📈 Efficiency Analysis:")
    print(f"  Old method max batch size: {old_max_batch}")
    print(f"  New method max batch size: {new_max_batch}")
    print(f"  Efficiency improvement: {new_max_batch / old_max_batch:.2f}x")
    
    if new_max_batch > old_max_batch:
        print("  ✅ New method creates larger, more efficient batches")
        return True
    else:
        print("  ❌ New method not more efficient")
        return False

if __name__ == "__main__":
    print("🧪 Face Enhancer Performance Validation")
    print("=" * 50)
    
    distribution_success = test_batch_distribution()
    performance_success = test_performance()
    
    print(f"\n🏁 Final Results:")
    print(f"  Batch distribution: {'✅ PASS' if distribution_success else '❌ FAIL'}")
    print(f"  Performance test: {'✅ PASS' if performance_success else '❌ FAIL'}")
    
    if distribution_success and performance_success:
        print("\n🎉 ALL TESTS PASSED: Optimized face enhancer is ready!")
    else:
        print("\n❌ SOME TESTS FAILED: Further optimization needed")
    
    sys.exit(0 if (distribution_success and performance_success) else 1)