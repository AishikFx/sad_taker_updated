#!/usr/bin/env python3
"""
Test script for SadTalker TTS integration
Demonstrates text-to-speech video generation
"""

import os
import sys
import argparse
import requests
import json

def test_direct_inference():
    """Test the direct inference method with TTS."""
    print("🧪 Testing Direct Inference with TTS...")
    
    # Add SadTalker to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    
    try:
        from inference import main
        
        # Create test arguments
        class TestArgs:
            def __init__(self):
                self.source_image = "./examples/source_image/full_body_1.png"
                self.driven_audio = None  # Will be generated from text
                self.input_text = "Hello! I am testing the SadTalker text-to-speech integration. This is a demonstration of Indian accent female voice."
                self.gender = "female"
                self.result_dir = "./results"
                self.checkpoint_dir = "./checkpoints"
                self.size = 256
                self.expression_scale = 1.0
                self.pose_style = 0
                self.enhancer = None
                self.background_enhancer = None
                self.device = "cuda"
                self.batch_size = 4
                self.optimization_preset = "balanced"
                self.preprocess = "crop"
                self.still = False
                self.face3dvis = False
                self.old_version = False
                self.verbose = False
                self.cpu = False
                self.profile = True  # Enable profiling
                
                # Optional parameters
                self.ref_eyeblink = None
                self.ref_pose = None
                self.input_yaw = None
                self.input_pitch = None
                self.input_roll = None
                
                # Advanced parameters with defaults
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
        
        args = TestArgs()
        
        # Check if required files exist
        if not os.path.exists(args.source_image):
            print(f"❌ Source image not found: {args.source_image}")
            return False
        
        if not os.path.exists(args.checkpoint_dir):
            print(f"❌ Checkpoint directory not found: {args.checkpoint_dir}")
            print("Please download SadTalker models first")
            return False
        
        print(f"📁 Source image: {args.source_image}")
        print(f"📝 Input text: {args.input_text}")
        print(f"👤 Gender: {args.gender}")
        
        # Run SadTalker with TTS
        output_path = main(args)
        
        if output_path and os.path.exists(output_path):
            print(f"✅ Video generated successfully: {output_path}")
            print(f"📁 File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
            return True
        else:
            print("❌ Video generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Direct inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_server():
    """Test the TTS API server."""
    print("\n🌐 Testing TTS API Server...")
    
    api_url = "http://localhost:8000"
    
    try:
        # Test health endpoint
        print("🏥 Testing health endpoint...")
        response = requests.get(f"{api_url}/health", timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ API server is healthy: {health_data}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
        
        # Test supported genders endpoint
        print("👥 Testing supported genders endpoint...")
        response = requests.get(f"{api_url}/supported-genders", timeout=10)
        
        if response.status_code == 200:
            genders_data = response.json()
            print(f"✅ Supported genders: {genders_data}")
        else:
            print(f"❌ Supported genders check failed: {response.status_code}")
        
        # Test video generation (if image file exists)
        test_image_path = "./examples/source_image/full_body_1.png"
        if os.path.exists(test_image_path):
            print("🎬 Testing video generation endpoint...")
            
            with open(test_image_path, 'rb') as image_file:
                files = {'image': image_file}
                data = {
                    'text': 'Hello! This is a test of the SadTalker TTS API integration.',
                    'gender': 'female'
                }
                
                response = requests.post(
                    f"{api_url}/generate-video-simple",
                    files=files,
                    data=data,
                    timeout=300  # 5 minutes timeout for video generation
                )
                
                if response.status_code == 200:
                    result_data = response.json()
                    print(f"✅ Video generation successful: {result_data}")
                    return True
                else:
                    print(f"❌ Video generation failed: {response.status_code}")
                    print(f"Response: {response.text}")
                    return False
        else:
            print(f"⚠️ Test image not found: {test_image_path}")
            print("Skipping video generation test")
            return True
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server")
        print("Make sure the server is running: python tts_api.py")
        return False
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test SadTalker TTS integration")
    parser.add_argument("--test-type", choices=["direct", "api", "both"], default="both",
                       help="Type of test to run")
    parser.add_argument("--text", default="Hello! I am testing the SadTalker text-to-speech integration.",
                       help="Text to use for TTS testing")
    
    args = parser.parse_args()
    
    print("🎬 SadTalker TTS Integration Test")
    print("=" * 50)
    
    success = True
    
    if args.test_type in ["direct", "both"]:
        success &= test_direct_inference()
    
    if args.test_type in ["api", "both"]:
        success &= test_api_server()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! SadTalker TTS integration is working.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        
        print("\n💡 Common issues and solutions:")
        print("1. Missing dependencies: pip install -r requirements_tts.txt")
        print("2. Missing models: Download SadTalker checkpoints")
        print("3. CUDA issues: Check GPU availability and memory")
        print("4. API server not running: python tts_api.py")
        
    return success

if __name__ == "__main__":
    main()