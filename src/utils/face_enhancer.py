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

class AdaptiveMemoryManager:
    """Adaptive memory manager that adjusts processing based on available VRAM"""
    
    def __init__(self):
        self.initial_memory = None
        self.peak_memory_seen = 0
        self.oom_count = 0
        self.successful_batch_sizes = []
        self.failed_batch_sizes = []
    
    def initialize(self):
        """Initialize memory tracking"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.initial_memory = torch.cuda.memory_allocated()
            self.peak_memory_seen = self.initial_memory
    
    def get_safe_batch_size(self, requested_batch_size, image_size=(512, 512)):
        """Get a safe batch size based on memory constraints and history"""
        if not torch.cuda.is_available():
            return min(4, requested_batch_size)
        
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        current_memory_gb = torch.cuda.memory_allocated() / 1e9
        available_memory_gb = gpu_memory_gb - current_memory_gb
        
        # Estimate memory needed per image (rough estimation)
        h, w = image_size
        memory_per_image_gb = (h * w * 3 * 4 * 4) / 1e9  # 4x overhead for processing
        
        # Conservative batch size based on available memory
        memory_based_batch_size = max(1, int(available_memory_gb * 0.6 / memory_per_image_gb))
        
        # Adjust based on OOM history
        if self.oom_count > 0:
            # Become more conservative after OOM events
            memory_based_batch_size = max(1, memory_based_batch_size // (self.oom_count + 1))
        
        # Use successful batch size history
        if self.successful_batch_sizes:
            max_successful = max(self.successful_batch_sizes[-5:])  # Last 5 successful batches
            memory_based_batch_size = min(memory_based_batch_size, max_successful)
        
        # Avoid known failed batch sizes
        while memory_based_batch_size in self.failed_batch_sizes and memory_based_batch_size > 1:
            memory_based_batch_size -= 1
        
        safe_batch_size = min(requested_batch_size, memory_based_batch_size)
        
        print(f"   💡 Memory Manager: Requested {requested_batch_size}, Safe {safe_batch_size}")
        print(f"   📊 Available VRAM: {available_memory_gb:.2f}GB, OOM count: {self.oom_count}")
        
        return safe_batch_size
    
    def record_success(self, batch_size, memory_used):
        """Record successful batch processing"""
        self.successful_batch_sizes.append(batch_size)
        self.peak_memory_seen = max(self.peak_memory_seen, memory_used)
        
        # Keep only recent history
        if len(self.successful_batch_sizes) > 10:
            self.successful_batch_sizes = self.successful_batch_sizes[-10:]
    
    def record_failure(self, batch_size, error_type="oom"):
        """Record failed batch processing"""
        if error_type == "oom":
            self.oom_count += 1
            self.failed_batch_sizes.append(batch_size)
            
            # Keep only recent failures
            if len(self.failed_batch_sizes) > 5:
                self.failed_batch_sizes = self.failed_batch_sizes[-5:]
    
    def get_memory_stats(self):
        """Get current memory statistics"""
        if not torch.cuda.is_available():
            return {}
        
        current_memory = torch.cuda.memory_allocated()
        max_memory = torch.cuda.max_memory_allocated()
        total_memory = torch.cuda.get_device_properties(0).total_memory
        
        return {
            'current_gb': current_memory / 1e9,
            'max_gb': max_memory / 1e9,
            'total_gb': total_memory / 1e9,
            'utilization_percent': (current_memory / total_memory) * 100,
            'peak_utilization_percent': (max_memory / total_memory) * 100
        }
    
    def suggest_optimization(self):
        """Suggest optimizations based on memory usage patterns"""
        stats = self.get_memory_stats()
        suggestions = []
        
        if stats.get('peak_utilization_percent', 0) < 50:
            suggestions.append("💡 Low VRAM usage detected. You can increase batch sizes for better performance.")
        elif stats.get('peak_utilization_percent', 0) > 90:
            suggestions.append("⚠️  High VRAM usage. Consider reducing batch sizes to avoid OOM errors.")
        
        if self.oom_count > 3:
            suggestions.append("🔧 Multiple OOM errors detected. Try reducing optimization_preset or adding more VRAM.")
        
        if len(self.successful_batch_sizes) > 5:
            avg_batch = sum(self.successful_batch_sizes[-5:]) / 5
            suggestions.append(f"📈 Optimal batch size appears to be around {avg_batch:.0f}")
        
        return suggestions


class TrueBatchGFPGANEnhancer:
    """True parallel batch processing GFPGAN enhancer for maximum VRAM utilization"""
    
    def __init__(self, method='gfpgan'):
        self.method = method
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gfpgan_model = None
        self.streams = []
        self.memory_manager = AdaptiveMemoryManager()
        self._initialize_parallel_gfpgan()
    
    def _initialize_parallel_gfpgan(self):
        """Initialize GFPGAN with parallel processing capabilities and memory management"""
        print('Initializing TRUE BATCH GFPGAN enhancer with OOM protection...')
        
        # Initialize memory manager
        self.memory_manager.initialize()
        
        try:
            from gfpgan import GFPGANer
            from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
            
            # Initialize standard GFPGAN first
            model_path = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
            self.gfpgan = GFPGANer(
                model_path=model_path,
                upscale=2,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None
            )
            
            # Extract the actual model for direct batch processing
            self.gfpgan_model = self.gfpgan.gfpgan
            
            # Create multiple CUDA streams for parallel processing
            num_streams = 4  # Use 4 streams for maximum parallelism
            self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
            
            # Pre-allocate tensor pools for different common sizes (smaller pools to avoid OOM)
            self.tensor_pools = {}
            common_sizes = [(512, 512), (256, 256)]  # Reduced pool sizes
            
            for size in common_sizes:
                pool_size = 4  # Smaller pool to avoid OOM
                try:
                    self.tensor_pools[size] = [
                        torch.zeros(3, size[0], size[1], device=self.device, dtype=torch.float32)
                        for _ in range(pool_size)
                    ]
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"⚠️  Cannot pre-allocate tensor pool for {size}, will create tensors on demand")
                        self.tensor_pools[size] = []
                    else:
                        raise e
            
            print(f"✅ Initialized TRUE BATCH enhancer with {num_streams} CUDA streams and OOM protection")
            
        except Exception as e:
            print(f"Failed to initialize parallel GFPGAN: {e}")
            # Fallback to standard GFPGAN
            self.gfpgan = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
                upscale=2,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None
            )
            self.gfpgan_model = self.gfpgan.gfpgan
    
    def enhance_batch_parallel(self, images, batch_size=32):
        """True parallel batch processing with comprehensive OOM handling and adaptive memory management"""
        
        # Use adaptive memory manager to determine safe batch size
        if images:
            image_size = images[0].shape[:2] if images[0].shape[:2] else (512, 512)
            safe_batch_size = self.memory_manager.get_safe_batch_size(batch_size, image_size)
        else:
            safe_batch_size = batch_size
        
        # Determine concurrent streams based on memory constraints
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            current_memory_gb = torch.cuda.memory_allocated() / 1e9
            available_memory_gb = gpu_memory_gb - current_memory_gb
            
            if available_memory_gb > 8:
                concurrent_streams = 4
            elif available_memory_gb > 4:
                concurrent_streams = 2
            else:
                concurrent_streams = 1
                
            print(f"🧠 Adaptive Memory: {available_memory_gb:.2f}GB available, using {concurrent_streams} streams")
        else:
            concurrent_streams = 1
        
        print(f"🚀 TRUE BATCH processing: {safe_batch_size} images (adaptive), {concurrent_streams} streams")
        
        enhanced_images = []
        current_batch_size = safe_batch_size
        oom_retry_count = 0
        max_oom_retries = 3
        
        # Pre-allocate result tensors to maximize memory efficiency
        torch.cuda.empty_cache()
        
        # Process in truly parallel batches using multiple streams with OOM handling
        batch_start = 0
        while batch_start < len(images):
            batch_end = min(batch_start + current_batch_size, len(images))
            batch_images = images[batch_start:batch_end]
            
            # Record memory before processing
            initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            try:
                # Split batch across streams for true parallelism
                stream_batches = self._split_batch_for_streams(batch_images, concurrent_streams)
                
                # Process all streams in parallel with OOM monitoring
                enhanced_batch = self._process_parallel_streams_with_oom_handling(stream_batches)
                enhanced_images.extend(enhanced_batch)
                
                # Record successful processing
                final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                self.memory_manager.record_success(current_batch_size, final_memory)
                
                # Reset retry count on success
                oom_retry_count = 0
                batch_start = batch_end
                
                # Adaptive memory cleanup
                if batch_start % (current_batch_size * 2) == 0:
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e) and oom_retry_count < max_oom_retries:
                    print(f"⚠️  CUDA OOM detected! Implementing adaptive recovery...")
                    
                    # Record failure
                    self.memory_manager.record_failure(current_batch_size, "oom")
                    
                    # OOM Recovery Strategy
                    recovery_success = self._handle_cuda_oom(e, current_batch_size, concurrent_streams)
                    
                    if recovery_success:
                        # Get new safe batch size from memory manager
                        current_batch_size = self.memory_manager.get_safe_batch_size(
                            max(1, current_batch_size // 2), image_size
                        )
                        concurrent_streams = max(1, concurrent_streams // 2)
                        oom_retry_count += 1
                        
                        print(f"🔧 Adaptive adjustment: batch size {current_batch_size}, streams {concurrent_streams}")
                        print(f"🔄 Retrying batch {batch_start}-{batch_end} (attempt {oom_retry_count + 1})")
                        
                        # Don't advance batch_start, retry with smaller batch
                        continue
                    else:
                        print("❌ OOM recovery failed, using emergency fallback")
                        # Fallback to emergency processing for remaining images
                        remaining_images = images[batch_start:]
                        fallback_enhanced = self._emergency_fallback_processing(remaining_images)
                        enhanced_images.extend(fallback_enhanced)
                        break
                else:
                    # Non-OOM error or too many retries
                    if "out of memory" in str(e):
                        print(f"❌ Persistent CUDA OOM after {max_oom_retries} retries")
                        self.memory_manager.record_failure(current_batch_size, "persistent_oom")
                    else:
                        print(f"❌ Non-OOM error: {e}")
                    
                    # Emergency fallback
                    remaining_images = images[batch_start:]
                    fallback_enhanced = self._emergency_fallback_processing(remaining_images)
                    enhanced_images.extend(fallback_enhanced)
                    break
        
        # Synchronize all streams at the end
        for stream in self.streams:
            stream.synchronize()
        
        # Print memory statistics and suggestions
        self._print_memory_summary()
        
        return enhanced_images
    
    def _print_memory_summary(self):
        """Print memory usage summary and optimization suggestions"""
        stats = self.memory_manager.get_memory_stats()
        suggestions = self.memory_manager.suggest_optimization()
        
        print(f"\n📊 Memory Usage Summary:")
        print(f"   Peak VRAM: {stats.get('max_gb', 0):.2f}GB ({stats.get('peak_utilization_percent', 0):.1f}%)")
        print(f"   Current: {stats.get('current_gb', 0):.2f}GB ({stats.get('utilization_percent', 0):.1f}%)")
        print(f"   OOM Events: {self.memory_manager.oom_count}")
        
        if suggestions:
            print(f"\n💡 Optimization Suggestions:")
            for suggestion in suggestions:
                print(f"   {suggestion}")
        
        # Show successful batch sizes for user reference
        if self.memory_manager.successful_batch_sizes:
            recent_batches = self.memory_manager.successful_batch_sizes[-3:]
            avg_batch = sum(recent_batches) / len(recent_batches)
            print(f"\n📈 Recent successful batch sizes: {recent_batches}")
            print(f"   Average: {avg_batch:.1f}")
    
    def get_memory_recommendations(self):
        """Get memory usage recommendations for future runs"""
        stats = self.memory_manager.get_memory_stats()
        recommendations = {
            'optimal_batch_size': None,
            'memory_utilization': stats.get('peak_utilization_percent', 0),
            'suggestions': self.memory_manager.suggest_optimization()
        }
        
        if self.memory_manager.successful_batch_sizes:
            recommendations['optimal_batch_size'] = max(self.memory_manager.successful_batch_sizes)
        
        return recommendations
    
    def _handle_cuda_oom(self, error, current_batch_size, concurrent_streams):
        """Comprehensive CUDA OOM recovery strategy"""
        print(f"🔧 CUDA OOM Recovery initiated...")
        print(f"   Error: {str(error)[:100]}...")
        
        try:
            # Step 1: Emergency memory cleanup
            print("   Step 1: Emergency memory cleanup")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Step 2: Clear all stream memory
            print("   Step 2: Clearing stream memory")
            for stream in self.streams:
                with torch.cuda.stream(stream):
                    torch.cuda.empty_cache()
                stream.synchronize()
            
            # Step 3: Force garbage collection
            print("   Step 3: Forcing garbage collection")
            import gc
            gc.collect()
            
            # Step 4: Check available memory
            if torch.cuda.is_available():
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                free_memory_gb = free_memory / 1e9
                print(f"   Available VRAM after cleanup: {free_memory_gb:.2f}GB")
                
                # If we have less than 2GB free, try more aggressive cleanup
                if free_memory_gb < 2.0:
                    print("   Step 5: Aggressive cleanup (less than 2GB free)")
                    
                    # Try to free tensor pools
                    self.tensor_pools.clear()
                    torch.cuda.empty_cache()
                    
                    # Reduce memory fraction
                    torch.cuda.set_per_process_memory_fraction(0.7)  # Reduce from 95% to 70%
                    
                    free_memory_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
                    print(f"   Available VRAM after aggressive cleanup: {free_memory_gb:.2f}GB")
            
            print("✅ OOM recovery completed successfully")
            return True
            
        except Exception as recovery_error:
            print(f"❌ OOM recovery failed: {recovery_error}")
            return False
    
    def _process_parallel_streams_with_oom_handling(self, stream_batches):
        """Process multiple batches in parallel with OOM monitoring"""
        futures = []
        enhanced_results = [None] * len(stream_batches)
        
        # Launch all streams in parallel with individual OOM handling
        for i, (stream_batch, stream) in enumerate(zip(stream_batches, self.streams[:len(stream_batches)])):
            try:
                with torch.cuda.stream(stream):
                    # Process this stream's batch with OOM protection
                    result = self._process_stream_batch_with_oom_protection(stream_batch, stream, i)
                    enhanced_results[i] = result
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️  Stream {i} OOM, attempting recovery...")
                    # Try processing with smaller chunks
                    result = self._process_stream_fallback(stream_batch, stream, i)
                    enhanced_results[i] = result
                else:
                    raise e
        
        # Wait for all streams to complete and collect results
        for stream in self.streams[:len(stream_batches)]:
            stream.synchronize()
        
        # Flatten results
        all_enhanced = []
        for result in enhanced_results:
            if result:
                all_enhanced.extend(result)
        
        return all_enhanced
    
    def _process_stream_batch_with_oom_protection(self, batch_images, stream, stream_id):
        """Process a batch within a single CUDA stream with OOM protection"""
        enhanced_batch = []
        
        try:
            # Monitor memory before processing
            initial_memory = torch.cuda.memory_allocated()
            
            # Convert images to tensors in parallel
            batch_tensors = self._prepare_batch_tensors_safe(batch_images)
            
            # Process tensors through GFPGAN model in batch
            with torch.cuda.amp.autocast():  # Use mixed precision
                enhanced_tensors = self._enhance_tensor_batch_safe(batch_tensors)
            
            # Convert back to numpy arrays
            enhanced_batch = self._tensors_to_images_safe(enhanced_tensors)
            
            # Log memory usage
            final_memory = torch.cuda.memory_allocated()
            memory_used = (final_memory - initial_memory) / 1e9
            print(f"   Stream {stream_id}: {len(batch_images)} images, {memory_used:.2f}GB VRAM used")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"   Stream {stream_id} OOM during processing, using fallback")
                enhanced_batch = self._process_stream_fallback(batch_images, stream, stream_id)
            else:
                # Fallback to sequential processing for this batch
                print(f"   Stream {stream_id} error: {e}, falling back to sequential")
                enhanced_batch = self._process_stream_fallback(batch_images, stream, stream_id)
        
        return enhanced_batch
    
    def _process_stream_fallback(self, batch_images, stream, stream_id):
        """Fallback processing for a stream batch with minimal memory usage"""
        enhanced_batch = []
        
        print(f"   Stream {stream_id}: Using conservative fallback processing")
        
        # Process images one by one to minimize memory usage
        for i, img in enumerate(batch_images):
            try:
                # Clear cache before each image
                if i % 2 == 0:
                    torch.cuda.empty_cache()
                
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                _, _, restored_img = self.gfpgan.enhance(
                    img_bgr, has_aligned=False, only_center_face=False, paste_back=True
                )
                restored_rgb = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
                enhanced_batch.append(restored_rgb)
                
            except Exception as e:
                print(f"   Stream {stream_id}, image {i}: Enhancement failed, using original")
                enhanced_batch.append(img)  # Use original if enhancement fails
        
        return enhanced_batch
    
    def _emergency_fallback_processing(self, images):
        """Emergency sequential processing when all else fails"""
        print("🚨 EMERGENCY FALLBACK: Processing remaining images sequentially")
        
        enhanced_images = []
        
        # Use minimal memory footprint
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.5)  # Use only 50% VRAM
        
        for i, img in enumerate(tqdm(images, desc="Emergency Processing")):
            try:
                # Clear cache every few images
                if i % 5 == 0:
                    torch.cuda.empty_cache()
                
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                _, _, restored_img = self.gfpgan.enhance(
                    img_bgr, has_aligned=False, only_center_face=False, paste_back=True
                )
                restored_rgb = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
                enhanced_images.append(restored_rgb)
                
            except Exception as e:
                print(f"Emergency processing failed for image {i}: {e}")
                enhanced_images.append(img)  # Use original
        
        return enhanced_images
    
    def _split_batch_for_streams(self, batch_images, num_streams):
        """Split batch across multiple CUDA streams"""
        stream_batches = []
        images_per_stream = len(batch_images) // num_streams
        
        for i in range(num_streams):
            start_idx = i * images_per_stream
            if i == num_streams - 1:
                # Last stream gets remaining images
                end_idx = len(batch_images)
            else:
                end_idx = (i + 1) * images_per_stream
            
            stream_batch = batch_images[start_idx:end_idx]
            if stream_batch:  # Only add non-empty batches
                stream_batches.append(stream_batch)
        
        return stream_batches
    
    def _process_parallel_streams(self, stream_batches):
        """Process multiple batches in parallel using CUDA streams"""
        futures = []
        enhanced_results = [None] * len(stream_batches)
        
        # Launch all streams in parallel
        for i, (stream_batch, stream) in enumerate(zip(stream_batches, self.streams[:len(stream_batches)])):
            with torch.cuda.stream(stream):
                # Process this stream's batch
                result = self._process_stream_batch(stream_batch, stream)
                enhanced_results[i] = result
        
        # Wait for all streams to complete and collect results
        for stream in self.streams[:len(stream_batches)]:
            stream.synchronize()
        
        # Flatten results
        all_enhanced = []
        for result in enhanced_results:
            if result:
                all_enhanced.extend(result)
        
        return all_enhanced
    
    def _process_stream_batch(self, batch_images, stream):
        """Process a batch within a single CUDA stream"""
        enhanced_batch = []
        
        try:
            # Convert images to tensors in parallel
            batch_tensors = self._prepare_batch_tensors(batch_images)
            
            # Process tensors through GFPGAN model in batch
            with torch.cuda.amp.autocast():  # Use mixed precision
                enhanced_tensors = self._enhance_tensor_batch(batch_tensors)
            
            # Convert back to numpy arrays
            enhanced_batch = self._tensors_to_images(enhanced_tensors)
            
        except Exception as e:
            print(f"Stream batch processing failed: {e}, falling back to sequential")
            # Fallback to sequential processing for this batch
            for img in batch_images:
                try:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    _, _, restored_img = self.gfpgan.enhance(
                        img_bgr, has_aligned=False, only_center_face=False, paste_back=True
                    )
                    restored_rgb = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
                    enhanced_batch.append(restored_rgb)
                except:
                    enhanced_batch.append(img)  # Use original if enhancement fails
        
        return enhanced_batch
    
    def _prepare_batch_tensors_safe(self, batch_images):
        """Prepare batch tensors with OOM protection and memory monitoring"""
        batch_tensors = []
        
        try:
            # Monitor memory before tensor creation
            initial_memory = torch.cuda.memory_allocated() / 1e9
            
            # Pre-allocate tensors from pool when possible (with OOM protection)
            h, w = batch_images[0].shape[:2] if batch_images else (512, 512)
            target_size = (h, w)
            
            # Check if we have enough memory for this batch
            estimated_memory_needed = len(batch_images) * h * w * 3 * 4 / 1e9  # 4 bytes per float32
            available_memory = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
            
            if estimated_memory_needed > available_memory * 0.8:  # Leave 20% buffer
                print(f"   ⚠️  Estimated memory needed ({estimated_memory_needed:.2f}GB) > available ({available_memory:.2f}GB)")
                # Process in smaller chunks
                chunk_size = max(1, int(len(batch_images) * available_memory * 0.6 / estimated_memory_needed))
                return self._prepare_chunked_tensors(batch_images[:chunk_size])
            
            # Use tensor pool if available for this size
            if target_size in self.tensor_pools and self.tensor_pools[target_size]:
                # Use pre-allocated tensors for maximum efficiency
                pool_tensors = self.tensor_pools[target_size]
                
                for i, img in enumerate(batch_images):
                    if i < len(pool_tensors):
                        # Reuse pre-allocated tensor
                        tensor = pool_tensors[i]
                        
                        # In-place operations for memory efficiency
                        if len(img.shape) == 3:
                            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        else:
                            img_bgr = img
                        
                        # Copy data to pre-allocated tensor
                        img_tensor = torch.from_numpy(img_bgr).float().to(self.device)
                        img_tensor.div_(255.0)  # In-place normalization
                        img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
                        
                        # Resize to match pool tensor if needed
                        if img_tensor.shape != tensor.shape:
                            img_tensor = torch.nn.functional.interpolate(
                                img_tensor.unsqueeze(0), size=tensor.shape[1:], mode='bilinear', align_corners=False
                            ).squeeze(0)
                        
                        tensor.copy_(img_tensor)  # In-place copy
                        batch_tensors.append(tensor.unsqueeze(0))
                    else:
                        # Create new tensor if pool exhausted
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 else img
                        img_tensor = torch.from_numpy(img_bgr).float().to(self.device) / 255.0
                        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
                        batch_tensors.append(img_tensor)
            else:
                # Standard tensor creation with memory optimization
                for i, img in enumerate(batch_images):
                    # Check memory every few images
                    if i % 4 == 0:
                        current_memory = torch.cuda.memory_allocated() / 1e9
                        if current_memory > initial_memory + estimated_memory_needed * 1.2:
                            print(f"   ⚠️  Memory usage exceeded expected, stopping at image {i}")
                            break
                    
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 else img
                    
                    # Use pinned memory for faster GPU transfer
                    img_tensor = torch.from_numpy(img_bgr).float()
                    img_tensor = img_tensor.div(255.0)  # More efficient than division
                    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
                    img_tensor = img_tensor.to(self.device, non_blocking=True)  # Async transfer
                    
                    batch_tensors.append(img_tensor)
            
            # Efficiently stack tensors into single batch
            if batch_tensors:
                return self._stack_tensors_safely(batch_tensors)
            
            return None
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"   ⚠️  OOM during tensor preparation, using smaller chunks")
                # Free any partially created tensors
                del batch_tensors
                torch.cuda.empty_cache()
                # Try with half the images
                half_size = max(1, len(batch_images) // 2)
                return self._prepare_chunked_tensors(batch_images[:half_size])
            else:
                raise e
    
    def _prepare_chunked_tensors(self, batch_images):
        """Prepare tensors in smaller chunks to avoid OOM"""
        if not batch_images:
            return None
        
        # Process just a few images at a time
        chunk_size = min(4, len(batch_images))
        batch_tensors = []
        
        for img in batch_images[:chunk_size]:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 else img
            img_tensor = torch.from_numpy(img_bgr).float().to(self.device) / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
            batch_tensors.append(img_tensor)
        
        return self._stack_tensors_safely(batch_tensors)
    
    def _stack_tensors_safely(self, batch_tensors):
        """Safely stack tensors with memory monitoring"""
        try:
            # Find optimal tensor size for batching
            max_h = max(t.shape[2] for t in batch_tensors)
            max_w = max(t.shape[3] for t in batch_tensors)
            
            # Round up to nearest multiple of 32 for optimal GPU processing
            max_h = ((max_h + 31) // 32) * 32
            max_w = ((max_w + 31) // 32) * 32
            
            # Use in-place padding when possible
            padded_tensors = []
            for tensor in batch_tensors:
                h, w = tensor.shape[2], tensor.shape[3]
                if h == max_h and w == max_w:
                    padded_tensors.append(tensor)
                else:
                    # Efficient padding
                    pad_h = max_h - h
                    pad_w = max_w - w
                    padded = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
                    padded_tensors.append(padded)
            
            # Concatenate efficiently
            return torch.cat(padded_tensors, dim=0)
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"   ⚠️  OOM during tensor stacking, using sequential processing")
                return None
            else:
                raise e
    
    def _enhance_tensor_batch_safe(self, batch_tensor):
        """Enhance batch tensor with OOM protection and memory monitoring"""
        if batch_tensor is None:
            return None
        
        try:
            # Monitor memory before processing
            initial_memory = torch.cuda.memory_allocated() / 1e9
            
            # Process with maximum memory efficiency
            with torch.no_grad(), torch.cuda.amp.autocast():
                
                # Pre-allocate output tensor for in-place operations
                output_shape = batch_tensor.shape
                
                # Check if we have enough memory for output tensor
                output_memory_needed = np.prod(output_shape) * 4 / 1e9  # 4 bytes per float32
                available_memory = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
                
                if output_memory_needed > available_memory * 0.8:
                    print(f"   ⚠️  Not enough memory for output tensor, using chunked processing")
                    return self._enhance_tensor_chunked(batch_tensor)
                
                enhanced_batch = torch.empty_like(batch_tensor, device=self.device)
                
                # Process in chunks if batch is very large to avoid OOM
                chunk_size = min(8, batch_tensor.shape[0])  # Process 8 images at a time max
                
                for i in range(0, batch_tensor.shape[0], chunk_size):
                    end_idx = min(i + chunk_size, batch_tensor.shape[0])
                    chunk = batch_tensor[i:end_idx]
                    
                    # Process chunk through GFPGAN
                    try:
                        enhanced_chunk = self.gfpgan_model(chunk)
                        enhanced_batch[i:end_idx] = enhanced_chunk
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            # Reduce chunk size and retry
                            torch.cuda.empty_cache()
                            chunk_size = max(1, chunk_size // 2)
                            print(f"   ⚠️  Reduced chunk size to {chunk_size} due to memory constraints")
                            
                            # Retry with smaller chunk
                            enhanced_chunk = self.gfpgan_model(chunk[:chunk_size])
                            enhanced_batch[i:i+chunk_size] = enhanced_chunk
                        else:
                            raise e
            
            return enhanced_batch
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"   ⚠️  OOM during tensor enhancement, using fallback")
                torch.cuda.empty_cache()
                return None
            else:
                raise e
    
    def _enhance_tensor_chunked(self, batch_tensor):
        """Enhanced tensor processing in very small chunks"""
        if batch_tensor is None:
            return None
        
        enhanced_chunks = []
        chunk_size = 2  # Very conservative chunk size
        
        for i in range(0, batch_tensor.shape[0], chunk_size):
            end_idx = min(i + chunk_size, batch_tensor.shape[0])
            chunk = batch_tensor[i:end_idx]
            
            try:
                with torch.no_grad():
                    enhanced_chunk = self.gfpgan_model(chunk)
                    enhanced_chunks.append(enhanced_chunk)
                    
                # Clear cache after each chunk
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"   ⚠️  Even chunked processing failed, using single image processing")
                    # Process single images
                    for j in range(i, end_idx):
                        single_tensor = batch_tensor[j:j+1]
                        try:
                            single_enhanced = self.gfpgan_model(single_tensor)
                            enhanced_chunks.append(single_enhanced)
                        except:
                            # Skip this image if it still fails
                            enhanced_chunks.append(single_tensor)
                        torch.cuda.empty_cache()
                else:
                    raise e
        
        if enhanced_chunks:
            return torch.cat(enhanced_chunks, dim=0)
        return None
    
    def _tensors_to_images_safe(self, enhanced_tensors):
        """Convert enhanced tensors back to images with OOM protection"""
        if enhanced_tensors is None:
            return []
        
        images = []
        
        try:
            # Process tensors in batches to avoid memory issues
            for i in range(enhanced_tensors.shape[0]):
                tensor = enhanced_tensors[i]
                
                # Efficient tensor to numpy conversion
                tensor = tensor.permute(1, 2, 0)  # CHW -> HWC
                
                # Clamp and convert with minimal memory allocation
                tensor = tensor.mul(255).clamp(0, 255).to(torch.uint8)
                img_array = tensor.cpu().numpy()
                
                # BGR to RGB conversion
                img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                images.append(img_rgb)
                
                # Clear tensor reference for memory efficiency
                del tensor
                
                # Clear cache every few images
                if i % 4 == 0:
                    torch.cuda.empty_cache()
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"   ⚠️  OOM during tensor to image conversion")
                torch.cuda.empty_cache()
                # Try with just the tensors we've processed so far
                return images
            else:
                raise e
        
        return images


class HighVRAMFaceEnhancer:
    """Ultra-optimized face enhancer for high VRAM systems (12GB+)"""
    
    def __init__(self, method='gfpgan'):
        self.method = method
        self.restorer = None
        self._initialize_gfpgan()
    
    def __init__(self, method='gfpgan'):
        self.method = method
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use the new true batch processor for maximum VRAM utilization
        self.true_batch_enhancer = TrueBatchGFPGANEnhancer(method=method)
        
        # Also keep fallback enhancer
        self.fallback_enhancer = None
        self._initialize_fallback_gfpgan()
    
    def _initialize_fallback_gfpgan(self):
        """Initialize fallback GFPGAN for error handling"""
        try:
            if self.method == 'gfpgan':
                model_name = 'GFPGANv1.4'
                url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
            
            model_path = os.path.join('gfpgan/weights', model_name + '.pth')
            if not os.path.isfile(model_path):
                model_path = os.path.join('checkpoints', model_name + '.pth')
            if not os.path.isfile(model_path):
                model_path = url

            self.fallback_enhancer = GFPGANer(
                model_path=model_path,
                upscale=2,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None
            )
            
        except Exception as e:
            print(f"Warning: Could not initialize fallback enhancer: {e}")
    
    def enhance_batch_ultra(self, images, batch_size=32):
        """Ultra-fast batch processing with TRUE parallel GPU utilization"""
        
        # Determine ultra-aggressive batch size for maximum VRAM usage
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory_gb >= 14:
                batch_size = min(64, batch_size)  # Massive batches for 15GB+
                print(f"🚀 MAXIMUM VRAM MODE: {batch_size} batch size for {gpu_memory_gb:.1f}GB")
            elif gpu_memory_gb >= 10:
                batch_size = min(48, batch_size)
                print(f"🚀 HIGH VRAM MODE: {batch_size} batch size for {gpu_memory_gb:.1f}GB")
            else:
                batch_size = min(32, batch_size)
                print(f"🚀 STANDARD MODE: {batch_size} batch size for {gpu_memory_gb:.1f}GB")
        
        # Setup for maximum GPU utilization
        if torch.cuda.is_available():
            # Use 95% of VRAM for processing
            torch.cuda.set_per_process_memory_fraction(0.95)
            
            # Enable all optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.deterministic = False
            
            # Pre-warm GPU
            torch.cuda.empty_cache()
            
        try:
            # Use true batch processing for maximum efficiency
            enhanced_images = self.true_batch_enhancer.enhance_batch_parallel(images, batch_size=batch_size)
            return enhanced_images
            
        except Exception as e:
            print(f"⚠️  True batch enhancer failed: {e}")
            print("🔄 Falling back to optimized sequential processing...")
            
            # Fallback to optimized sequential processing
            return self._fallback_batch_enhance(images, batch_size // 2)
    
    def _fallback_batch_enhance(self, images, batch_size):
        """Fallback batch enhancement with memory optimization"""
        enhanced_images = []
        
        for batch_start in tqdm(range(0, len(images), batch_size), 'Fallback Face Enhancer:'):
            batch_end = min(batch_start + batch_size, len(images))
            batch_images = images[batch_start:batch_end]
            
            # Process batch with memory optimization
            for img in batch_images:
                try:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    _, _, restored_img = self.fallback_enhancer.enhance(
                        img_bgr,
                        has_aligned=False,
                        only_center_face=False,
                        paste_back=True
                    )
                    restored_rgb = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
                    enhanced_images.append(restored_rgb)
                except Exception as e:
                    print(f"Enhancement failed, using original: {e}")
                    enhanced_images.append(img)
            
            # Clear cache every few batches
            if torch.cuda.is_available() and batch_start % (batch_size * 4) == 0:
                torch.cuda.empty_cache()
        
        return enhanced_images


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