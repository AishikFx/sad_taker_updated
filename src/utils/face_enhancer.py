# src/utils/fast_face_enhancer.py
import os
import torch 
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import multiprocessing

from gfpgan import GFPGANer
from src.utils.videoio import load_video_to_cv2


class FastGeneratorWithLen(object):
    """ Optimized generator with length """
    def __init__(self, gen, length):
        self.gen = gen
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.gen

class LightweightEnhancer:
    """Lightweight alternative to GFPGAN for extreme speed optimization"""
    
    def __init__(self):
        # Simple sharpening kernel
        self.sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        # Noise reduction kernel
        self.denoise_kernel = np.ones((3,3), np.float32) / 9
    
    def enhance(self, image):
        """Fast enhancement using basic image processing"""
        # Convert to float for processing
        img_float = image.astype(np.float32) / 255.0
        
        # Apply slight sharpening
        sharpened = cv2.filter2D(img_float, -1, self.sharpen_kernel)
        sharpened = np.clip(sharpened, 0, 1)
        
        # Enhance contrast slightly
        enhanced = cv2.convertScaleAbs(sharpened * 255, alpha=1.1, beta=5)
        
        # Simple color correction
        enhanced = cv2.bilateralFilter(enhanced, 5, 75, 75)
        
        return enhanced

class OptimizedFaceEnhancer:
    def __init__(self, method='gfpgan', optimization_level="medium"):
        self.method = method
        self.optimization_level = optimization_level
        self.restorer = None
        self.lightweight_enhancer = LightweightEnhancer()
        
        # Initialize based on optimization level
        if optimization_level == "extreme":
            print("Using lightweight enhancement for extreme speed")
            # Don't initialize GFPGAN for extreme optimization
        else:
            self._initialize_gfpgan()
    
    def _initialize_gfpgan(self):
        """Initialize GFPGAN with optimizations"""
        print('Initializing optimized face enhancer...')
        
        if self.method == 'gfpgan':
            arch = 'clean'
            channel_multiplier = 2
            model_name = 'GFPGANv1.4'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
        elif self.method == 'RestoreFormer':
            arch = 'RestoreFormer'
            channel_multiplier = 2
            model_name = 'RestoreFormer'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
        else:
            raise ValueError(f'Unsupported method {self.method}')

        # Background upsampler optimization
        bg_upsampler = None
        if self.optimization_level in ["low", "medium"] and torch.cuda.is_available():
            try:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                bg_upsampler = RealESRGANer(
                    scale=2,
                    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                    model=model,
                    tile=400,
                    tile_pad=10,
                    pre_pad=0,
                    half=True)
            except:
                bg_upsampler = None

        # Model path handling
        model_path = os.path.join('gfpgan/weights', model_name + '.pth')
        if not os.path.isfile(model_path):
            model_path = os.path.join('checkpoints', model_name + '.pth')
        if not os.path.isfile(model_path):
            model_path = url

        self.restorer = GFPGANer(
            model_path=model_path,
            upscale=2,
            arch=arch,
            channel_multiplier=channel_multiplier,
            bg_upsampler=bg_upsampler
        )

    def enhance_batch(self, images, batch_size=4):
        """Enhanced batch processing with memory optimization"""
        
        if self.optimization_level == "extreme":
            # Use lightweight enhancement
            return self._lightweight_batch_enhance(images, batch_size)
        
        if not isinstance(images, list) and os.path.isfile(images):
            images = load_video_to_cv2(images)
        
        enhanced_images = []
        
        # Determine optimal batch size based on GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            if gpu_memory > 8e9:  # 8GB+
                batch_size = min(8, batch_size)
            elif gpu_memory > 4e9:  # 4GB+
                batch_size = min(4, batch_size)
            else:
                batch_size = min(2, batch_size)
        
        # Process in batches with parallel processing
        if self.optimization_level in ["high", "extreme"] and len(images) > batch_size:
            return self._parallel_batch_enhance(images, batch_size)
        else:
            return self._sequential_batch_enhance(images, batch_size)
    
    def _lightweight_batch_enhance(self, images, batch_size):
        """Ultra-fast lightweight enhancement"""
        def enhance_batch_lightweight(batch_images):
            return [self.lightweight_enhancer.enhance(img) for img in batch_images]
        
        enhanced_images = []
        num_workers = min(4, multiprocessing.cpu_count())
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for batch_start in range(0, len(images), batch_size):
                batch_end = min(batch_start + batch_size, len(images))
                batch = images[batch_start:batch_end]
                
                # Convert BGR to RGB if needed
                batch_rgb = []
                for img in batch:
                    if len(img.shape) == 3:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[2] == 3 else img
                    else:
                        img_rgb = img
                    batch_rgb.append(img_rgb)
                
                future = executor.submit(enhance_batch_lightweight, batch_rgb)
                batch_enhanced = future.result()
                enhanced_images.extend(batch_enhanced)
        
        return enhanced_images
    
    def _parallel_batch_enhance(self, images, batch_size):
        """Parallel batch enhancement with GFPGAN"""
        enhanced_images = []
        num_workers = 2  # Limit to 2 workers to avoid GPU memory issues
        
        def enhance_single_batch(batch_images):
            batch_results = []
            for img in batch_images:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                try:
                    _, _, restored_img = self.restorer.enhance(
                        img_bgr,
                        has_aligned=False,
                        only_center_face=False,
                        paste_back=True
                    )
                    restored_rgb = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
                    batch_results.append(restored_rgb)
                except Exception as e:
                    print(f"Enhancement failed for image, using original: {e}")
                    batch_results.append(img)
            return batch_results
        
        # Process batches sequentially to avoid GPU memory conflicts
        for batch_start in tqdm(range(0, len(images), batch_size), 'Fast Face Enhancer:'):
            batch_end = min(batch_start + batch_size, len(images))
            batch = images[batch_start:batch_end]
            
            batch_enhanced = enhance_single_batch(batch)
            enhanced_images.extend(batch_enhanced)
            
            # Clear GPU cache periodically
            if torch.cuda.is_available() and batch_start % (batch_size * 4) == 0:
                torch.cuda.empty_cache()
        
        return enhanced_images
    
    def _sequential_batch_enhance(self, images, batch_size):
        """Standard sequential batch enhancement"""
        enhanced_images = []
        
        for batch_start in tqdm(range(0, len(images), batch_size), 'Face Enhancer (batch):'):
            batch_end = min(batch_start + batch_size, len(images))
            
            for idx in range(batch_start, batch_end):
                img = cv2.cvtColor(images[idx], cv2.COLOR_RGB2BGR)
                
                try:
                    _, _, restored_img = self.restorer.enhance(
                        img,
                        has_aligned=False,
                        only_center_face=False,
                        paste_back=True
                    )
                    restored_rgb = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
                    enhanced_images.append(restored_rgb)
                except Exception as e:
                    print(f"Enhancement failed for image {idx}, using original: {e}")
                    enhanced_images.append(images[idx])
            
            # Periodic cleanup
            if torch.cuda.is_available() and batch_start % (batch_size * 2) == 0:
                torch.cuda.empty_cache()
        
        return enhanced_images


def fast_enhancer_generator_with_len(images, method='gfpgan', bg_upsampler='realesrgan', 
                                   batch_size=4, optimization_level="medium"):
    """Fast enhanced generator with length support"""
    
    if os.path.isfile(images):
        images = load_video_to_cv2(images)
    
    enhancer = OptimizedFaceEnhancer(method=method, optimization_level=optimization_level)
    enhanced_images = enhancer.enhance_batch(images, batch_size=batch_size)
    
    def enhanced_generator():
        for img in enhanced_images:
            yield img
    
    gen_with_len = FastGeneratorWithLen(enhanced_generator(), len(enhanced_images))
    return gen_with_len


def fast_enhancer_list(images, method='gfpgan', bg_upsampler='realesrgan', 
                      batch_size=4, optimization_level="medium"):
    """Fast enhanced list generation"""
    
    if os.path.isfile(images):
        images = load_video_to_cv2(images)
    
    enhancer = OptimizedFaceEnhancer(method=method, optimization_level=optimization_level)
    return enhancer.enhance_batch(images, batch_size=batch_size)


# Memory-optimized streaming enhancer
class StreamingEnhancer:
    """Memory-efficient streaming face enhancer"""
    
    def __init__(self, method='gfpgan', optimization_level="medium", max_memory_mb=1024):
        self.enhancer = OptimizedFaceEnhancer(method=method, optimization_level=optimization_level)
        self.max_memory_mb = max_memory_mb
        self.frame_buffer = Queue(maxsize=32)  # Buffer for frames
    
    def enhance_stream(self, video_path, batch_size=4):
        """Stream-process video frames to minimize memory usage"""
        
        def frame_producer():
            """Producer thread to load frames"""
            cap = cv2.VideoCapture(video_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_buffer.put(frame_rgb)
            cap.release()
            self.frame_buffer.put(None)  # Signal end
        
        def frame_consumer():
            """Consumer thread to enhance frames"""
            batch = []
            while True:
                frame = self.frame_buffer.get()
                if frame is None:
                    # Process remaining frames
                    if batch:
                        if self.enhancer.optimization_level == "extreme":
                            enhanced_batch = self.enhancer._lightweight_batch_enhance(batch, len(batch))
                        else:
                            enhanced_batch = self.enhancer._sequential_batch_enhance(batch, len(batch))
                        for enhanced_frame in enhanced_batch:
                            yield enhanced_frame
                    break
                
                batch.append(frame)
                if len(batch) >= batch_size:
                    # Process batch
                    if self.enhancer.optimization_level == "extreme":
                        enhanced_batch = self.enhancer._lightweight_batch_enhance(batch, batch_size)
                    else:
                        enhanced_batch = self.enhancer._sequential_batch_enhance(batch, batch_size)
                    
                    for enhanced_frame in enhanced_batch:
                        yield enhanced_frame
                    batch = []
        
        # Start producer thread
        import threading
        producer_thread = threading.Thread(target=frame_producer)
        producer_thread.daemon = True
        producer_thread.start()
        
        # Return consumer generator
        return frame_consumer()


# GPU memory management utilities
class GPUMemoryManager:
    """Manage GPU memory for optimal performance"""
    
    @staticmethod
    def get_optimal_batch_size():
        """Determine optimal batch size based on available GPU memory"""
        if not torch.cuda.is_available():
            return 1
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_gb = gpu_memory / 1e9
        
        if gpu_memory_gb >= 12:
            return 8
        elif gpu_memory_gb >= 8:
            return 6
        elif gpu_memory_gb >= 6:
            return 4
        elif gpu_memory_gb >= 4:
            return 2
        else:
            return 1
    
    @staticmethod
    def setup_memory_optimization():
        """Setup memory optimizations"""
        if torch.cuda.is_available():
            # Enable memory optimization
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Set memory fraction
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(0.8)
    
    @staticmethod
    def cleanup_gpu_memory():
        """Aggressive GPU memory cleanup"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# Quality vs Speed presets
class EnhancementPresets:
    """Predefined presets for different quality/speed tradeoffs"""
    
    PRESETS = {
        "ultra_fast": {
            "optimization_level": "extreme",
            "method": "lightweight",
            "batch_size": 16,
            "skip_bg_upsampler": True,
            "description": "Fastest processing, basic enhancement only"
        },
        "fast": {
            "optimization_level": "high", 
            "method": "gfpgan",
            "batch_size": 8,
            "skip_bg_upsampler": True,
            "description": "Fast processing with GFPGAN, no background upsampling"
        },
        "balanced": {
            "optimization_level": "medium",
            "method": "gfpgan", 
            "batch_size": 4,
            "skip_bg_upsampler": False,
            "description": "Balanced quality and speed"
        },
        "quality": {
            "optimization_level": "low",
            "method": "RestoreFormer",
            "batch_size": 2,
            "skip_bg_upsampler": False,
            "description": "Best quality, slower processing"
        }
    }
    
    @classmethod
    def get_preset(cls, preset_name):
        """Get enhancement preset configuration"""
        if preset_name not in cls.PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(cls.PRESETS.keys())}")
        return cls.PRESETS[preset_name].copy()
    
    @classmethod
    def list_presets(cls):
        """List available presets with descriptions"""
        for name, config in cls.PRESETS.items():
            print(f"{name}: {config['description']}")


# Usage example and benchmark function
def benchmark_enhancement_methods(video_path, num_frames=50):
    """Benchmark different enhancement methods"""
    import time
    
    # Load test frames
    frames = load_video_to_cv2(video_path)[:num_frames]
    
    results = {}
    
    for preset_name in EnhancementPresets.PRESETS.keys():
        print(f"\nBenchmarking {preset_name}...")
        preset = EnhancementPresets.get_preset(preset_name)
        
        start_time = time.time()
        
        if preset_name == "ultra_fast":
            enhancer = OptimizedFaceEnhancer(optimization_level="extreme")
            enhanced = enhancer._lightweight_batch_enhance(frames, preset["batch_size"])
        else:
            enhanced = fast_enhancer_list(
                frames, 
                method=preset["method"],
                batch_size=preset["batch_size"],
                optimization_level=preset["optimization_level"]
            )
        
        end_time = time.time()
        processing_time = end_time - start_time
        fps = len(frames) / processing_time
        
        results[preset_name] = {
            "time": processing_time,
            "fps": fps,
            "frames": len(enhanced)
        }
        
        print(f"{preset_name}: {processing_time:.2f}s, {fps:.2f} FPS")
    
    return results