#!/usr/bin/env python3
"""
Test script to verify the tensor shape fix for face enhancement
"""

import sys
import os
sys.path.append('src')

import torch
import cv2
import numpy as np
from utils.face_enhancer import create_optimized_enhancer

def test_tensor_shape_fix():
    """Test that the tensor shape issue is resolved"""
    print("🧪 Testing tensor shape fix for face enhancement...")
    
    # Create test images
    test_images = []
    for i in range(6):  # Create 6 test images (similar to the error case)
        # Create random 512x512 RGB image
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        test_images.append(img)
    
    print(f"Created {len(test_images)} test images of shape {test_images[0].shape}")
    
    # Test with optimized enhancer
    try:
        enhancer = create_optimized_enhancer(method='gfpgan', optimization_level="medium")
        print("✅ Enhancer created successfully")
        
        # Test batch enhancement
        print("🚀 Testing batch enhancement with tensor shape fix...")
        enhanced_images = enhancer.enhance_batch(test_images, batch_size=6)
        
        print(f"✅ Enhancement completed successfully!")
        print(f"   Input: {len(test_images)} images")
        print(f"   Output: {len(enhanced_images)} images")
        print(f"   Shape: {enhanced_images[0].shape if enhanced_images else 'None'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_tensor_shape_fix()
    if success:
        print("\n🎉 Tensor shape fix test PASSED!")
        print("The matrix multiplication error should be resolved.")
    else:
        print("\n💥 Tensor shape fix test FAILED!")
        print("The issue may need further investigation.")