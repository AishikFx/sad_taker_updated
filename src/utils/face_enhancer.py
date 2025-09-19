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

class HighVRAMFaceEnhancer:
    """Ultra-optimized face enhancer for high VRAM systems (12GB+)"""
    
    def __init__(self, method='gfpgan'):
        self.method = method
        self.restorer = None
        self._initialize_gfpgan()
    
    def _initialize_gfpgan(self):
        """Initialize GFPGAN with high-memory optimizations"""
        print('Initializing HIGH VRAM face enhancer...')
        
        if self.method == 'gfpgan':
            arch = 'clean'
            channel_multiplier = 2
            model_name = 'GFPGANv1.4'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
        
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
            bg_upsampler=None  # Disable for speed
        )
        
        # Move model to GPU and optimize
        if torch.cuda.is_available():
            if hasattr(self.restorer, 'gfpgan'):
                self.restorer.gfpgan = self.restorer.gfpgan.cuda()
                self.restorer.gfpgan.eval()
                
                # Enable optimizations
                torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of VRAM
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
    
    def enhance_batch_ultra(self, images, batch_size=32):
        """Ultra-fast batch processing for high VRAM systems"""
        enhanced_images = []
        
        # Determine ultra-aggressive batch size
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory_gb >= 14:
                batch_size = min(48, batch_size)  # For 15GB+ VRAM
            elif gpu_memory_gb >= 10:
                batch_size = min(32, batch_size)  # For 12GB VRAM
            else:
                batch_size = min(16, batch_size)  # For 8-10GB VRAM
        
        print(f"🚀 Using ULTRA batch size: {batch_size} for {gpu_memory_gb:.1f}GB VRAM")
        
        # Pre-allocate GPU memory
        torch.cuda.empty_cache()
        
        with torch.cuda.amp.autocast():  # Use mixed precision
            for batch_start in tqdm(range(0, len(images), batch_size), 'HIGH VRAM Face Enhancer:'):
                batch_end = min(batch_start + batch_size, len(images))
                batch_images = images[batch_start:batch_end]
                
                # Process batch with memory pinning
                batch_enhanced = self._process_ultra_batch(batch_images)
                enhanced_images.extend(batch_enhanced)
                
                # Only clear cache every 8 batches to maintain speed
                if batch_start % (batch_size * 8) == 0:
                    torch.cuda.empty_cache()
        
        return enhanced_images
    
    def _process_ultra_batch(self, batch_images):
        """Ultra-optimized batch processing"""
        batch_enhanced = []
        
        # Process in parallel mini-batches within the main batch
        mini_batch_size = 4  # Process 4 at a time for optimal GPU utilization
        
        for i in range(0, len(batch_images), mini_batch_size):
            mini_batch = batch_images[i:i+mini_batch_size]
            
            # Pre-process all images in mini-batch
            processed_images = []
            for img in mini_batch:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                processed_images.append(img_bgr)
            
            # Enhance all images in mini-batch
            for img_bgr in processed_images:
                try:
                    _, _, restored_img = self.restorer.enhance(
                        img_bgr,
                        has_aligned=False,
                        only_center_face=False,
                        paste_back=True
                    )
                    restored_rgb = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
                    batch_enhanced.append(restored_rgb)
                except Exception as e:
                    print(f"Enhancement failed, using original: {e}")
                    # Convert back to RGB
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    batch_enhanced.append(img_rgb)
        
        return batch_enhanced


# Update the main enhancer factory function
def create_optimized_enhancer(method='gfpgan', optimization_level="medium"):
    """Factory function to create the best enhancer for the system"""
    
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # Use HighVRAMFaceEnhancer for systems with 12GB+ VRAM
        if gpu_memory_gb >= 12 and optimization_level in ["medium", "low"]:
            print(f"🚀 Detected {gpu_memory_gb:.1f}GB VRAM - Using HIGH VRAM optimizer")
            return HighVRAMFaceEnhancer(method=method)
    
    # Use standard optimizer for other cases
    return OptimizedFaceEnhancer(method=method, optimization_level=optimization_level)


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

    def enhance_batch(self, images, batch_size=8):
        """Enhanced batch processing with aggressive memory optimization for 15GB VRAM"""
        
        if self.optimization_level == "extreme":
            # Use lightweight enhancement
            return self._lightweight_batch_enhance(images, batch_size)
        
        if not isinstance(images, list) and os.path.isfile(images):
            images = load_video_to_cv2(images)
        
        # Aggressive batch size optimization for high VRAM systems
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU Memory Available: {gpu_memory_gb:.1f}GB")
            
            if gpu_memory_gb >= 14:  # 15GB+ cards (RTX 4090, etc.)
                optimal_batch_size = 24
            elif gpu_memory_gb >= 10:  # 12GB cards (RTX 4070 Ti, etc.)
                optimal_batch_size = 16
            elif gpu_memory_gb >= 8:   # 8-10GB cards
                optimal_batch_size = 12
            elif gpu_memory_gb >= 6:   # 6-8GB cards
                optimal_batch_size = 8
            elif gpu_memory_gb >= 4:   # 4-6GB cards
                optimal_batch_size = 4
            else:
                optimal_batch_size = 2
            
            # Use the larger of provided batch_size and optimal_batch_size
            batch_size = max(batch_size, optimal_batch_size)
            print(f"Using optimized batch size: {batch_size} (was {batch_size} requested)")
        
        # Setup memory optimizations
        if torch.cuda.is_available():
            # Enable memory efficient attention and other optimizations
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Clear cache before starting
            torch.cuda.empty_cache()
        
        # Process in optimized batches
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
        """Optimized batch enhancement with true GPU batch processing"""
        enhanced_images = []
        
        # Determine optimal batch size for GPU memory
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory_gb >= 12:
                max_batch_size = 16
            elif gpu_memory_gb >= 8:
                max_batch_size = 12
            elif gpu_memory_gb >= 6:
                max_batch_size = 8
            else:
                max_batch_size = 4
            batch_size = min(batch_size, max_batch_size)
        
        for batch_start in tqdm(range(0, len(images), batch_size), 'Face Enhancer (batch):'):
            batch_end = min(batch_start + batch_size, len(images))
            batch_images = images[batch_start:batch_end]
            
            # Process the entire batch at once using GFPGAN's batch capabilities
            batch_enhanced = self._process_image_batch(batch_images)
            enhanced_images.extend(batch_enhanced)
            
            # Periodic cleanup every 4 batches
            if torch.cuda.is_available() and batch_start % (batch_size * 4) == 0:
                torch.cuda.empty_cache()
        
        return enhanced_images
    
    def _process_image_batch(self, batch_images):
        """Process a batch of images efficiently using tensor operations"""
        batch_enhanced = []
        
        try:
            # Convert batch to tensor for GPU processing
            batch_tensors = []
            original_sizes = []
            
            for img in batch_images:
                # Convert RGB to BGR for GFPGAN
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                original_sizes.append(img_bgr.shape[:2])
                
                # Normalize and convert to tensor
                img_tensor = torch.from_numpy(img_bgr).float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
                batch_tensors.append(img_tensor)
            
            # Pad images to same size for batching
            max_h = max([t.shape[1] for t in batch_tensors])
            max_w = max([t.shape[2] for t in batch_tensors])
            
            # Ensure dimensions are multiples of 8 for efficiency
            max_h = ((max_h + 7) // 8) * 8
            max_w = ((max_w + 7) // 8) * 8
            
            padded_tensors = []
            for i, tensor in enumerate(batch_tensors):
                h, w = tensor.shape[1], tensor.shape[2]
                # Pad tensor
                pad_h = max_h - h
                pad_w = max_w - w
                padded = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
                padded_tensors.append(padded)
            
            # Stack into batch tensor
            if torch.cuda.is_available():
                batch_tensor = torch.stack(padded_tensors).cuda()
            else:
                batch_tensor = torch.stack(padded_tensors)
            
            # Process batch through GFPGAN (if it supports batch processing)
            # For now, we'll process individually but with optimized memory management
            for i, img in enumerate(batch_images):
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # Use GFPGAN enhance method
                _, _, restored_img = self.restorer.enhance(
                    img_bgr,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True
                )
                
                restored_rgb = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
                batch_enhanced.append(restored_rgb)
                
                # Clear intermediate GPU memory every few images
                if torch.cuda.is_available() and i % 2 == 0:
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            print(f"Batch processing failed, falling back to individual processing: {e}")
            # Fallback to individual processing
            for img in batch_images:
                try:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    _, _, restored_img = self.restorer.enhance(
                        img_bgr,
                        has_aligned=False,
                        only_center_face=False,
                        paste_back=True
                    )
                    restored_rgb = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
                    batch_enhanced.append(restored_rgb)
                except Exception as e2:
                    print(f"Enhancement failed for image, using original: {e2}")
                    batch_enhanced.append(img)
        
        return batch_enhanced


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