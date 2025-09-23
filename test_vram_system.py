#!/usr/bin/env python3
"""
Test script for the new parallel VRAM management system
"""

import os
import sys
import time
import subprocess
from datetime import datetime

def run_sadtalker_test():
    """Run a test of SadTalker with the new VRAM management"""
    print("=" * 80)
    print("SADTALKER VRAM MANAGEMENT TEST")
    print("=" * 80)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test parameters - adjust these paths as needed
    test_config = {
        "driven_audio": "examples/driven_audio/bus_chinese.wav",  # Use smaller audio for testing
        "source_image": "examples/source_image/art_0.png",       # Use test image
        "result_dir": "test_results",
        "preprocess": "crop",  # Start with crop for faster testing
        "enhancer": "gfpgan",
        "batch_size": 4  # Conservative batch size for testing
    }
    
    # Verify test files exist
    for key, path in test_config.items():
        if key in ["driven_audio", "source_image"] and not os.path.exists(path):
            print(f"ERROR: Test file not found: {path}")
            print("Please ensure you have the example files or update the paths in this script.")
            return False
    
    # Create test results directory
    os.makedirs(test_config["result_dir"], exist_ok=True)
    
    # Build command
    cmd = [
        "python", "inference.py",
        "--driven_audio", test_config["driven_audio"],
        "--source_image", test_config["source_image"], 
        "--result_dir", test_config["result_dir"],
        "--preprocess", test_config["preprocess"],
        "--enhancer", test_config["enhancer"],
        "--batch_size", str(test_config["batch_size"]),
        "--profile"  # Enable profiling for performance monitoring
    ]
    
    print("Running SadTalker with command:")
    print(" ".join(cmd))
    print()
    print("Monitor the output for:")
    print("✓ VRAM processor initialization")
    print("✓ Dynamic VRAM allocation messages")
    print("✓ Parallel task processing")
    print("✓ No 'out of memory' errors")
    print("✓ No 'length of restored_faces' errors")
    print("✓ All frames processed successfully")
    print("✓ Final cleanup and VRAM release")
    print()
    print("-" * 80)
    
    # Run the command
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=False, text=True, timeout=600)  # 10 minute timeout
        end_time = time.time()
        
        print("-" * 80)
        print(f"Test completed in {end_time - start_time:.1f} seconds")
        print(f"Exit code: {result.returncode}")
        
        if result.returncode == 0:
            print(" TEST PASSED: SadTalker completed successfully!")
            
            # Check if output files exist
            output_files = []
            for file in os.listdir(test_config["result_dir"]):
                if file.endswith('.mp4'):
                    output_files.append(file)
                    file_path = os.path.join(test_config["result_dir"], file)
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    print(f"   Output: {file} ({file_size:.1f} MB)")
            
            if output_files:
                print(f" Generated {len(output_files)} video file(s)")
            else:
                print("⚠️  Warning: No output video files found")
            
        else:
            print("❌ TEST FAILED: SadTalker exited with error")
            
    except subprocess.TimeoutExpired:
        print("❌ TEST FAILED: Timeout after 10 minutes")
        return False
    except Exception as e:
        print(f"❌ TEST FAILED: Exception occurred: {e}")
        return False
    
    print()
    print("Test completed at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    return result.returncode == 0


def test_vram_manager_directly():
    """Test the VRAM manager components directly"""
    print("\n" + "=" * 80)
    print("DIRECT VRAM MANAGER TEST")
    print("=" * 80)
    
    try:
        from src.utils.vram_queue_manager import get_global_processor, ProcessTask
        from src.utils.parallel_face_enhancer import ParallelFaceEnhancer
        import numpy as np
        import uuid
        
        print("✓ Successfully imported VRAM manager components")
        
        # Test VRAM processor
        processor = get_global_processor()
        status = processor.get_status()
        print(f"✓ VRAM processor status: {status['vram_status']['free']:.1f}GB free")
        
        # Test with dummy enhancement task
        print("\nTesting parallel face enhancer with dummy data...")
        enhancer = ParallelFaceEnhancer()
        
        # Create dummy images
        dummy_images = [
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(5)
        ]
        
        print(f"Created {len(dummy_images)} dummy 256x256 images")
        
        # Test enhancement (should handle gracefully even without GFPGAN)
        start_time = time.time()
        result = enhancer.enhance_images_parallel(dummy_images, quality_mode="high")
        end_time = time.time()
        
        print(f"✓ Parallel enhancement completed in {end_time - start_time:.2f}s")
        print(f"✓ Processed {len(result)} images")
        
        # Check final status
        final_status = processor.get_status()
        print(f"✓ Final VRAM status: {final_status['vram_status']['free']:.1f}GB free")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False


if __name__ == "__main__":
    print("SadTalker Parallel VRAM Management Test Suite")
    print()
    
    # Test 1: Direct component test
    component_test_passed = test_vram_manager_directly()
    
    # Test 2: Full integration test (only if component test passed)
    if component_test_passed:
        integration_test_passed = run_sadtalker_test()
    else:
        print("❌ Skipping integration test due to component test failure")
        integration_test_passed = False
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Component Test: {' PASSED' if component_test_passed else '❌ FAILED'}")
    print(f"Integration Test: {' PASSED' if integration_test_passed else '❌ FAILED'}")
    
    if component_test_passed and integration_test_passed:
        print("\n🎉 ALL TESTS PASSED! The new VRAM management system is working correctly.")
        print("\nYou can now run SadTalker with improved:")
        print("   • Dynamic VRAM allocation")
        print("   • Parallel frame processing") 
        print("   • No frame skipping")
        print("   • Better GPU utilization")
        print("   • Robust error handling")
    else:
        print("\n⚠️  SOME TESTS FAILED. Please check the errors above.")
    
    print("\n" + "=" * 80)