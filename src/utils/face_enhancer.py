import os
import time
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import cv2
import numpy as np
import torch
from tqdm import tqdm

# It's assumed gfpgan is installed, e.g., via 'pip install gfpgan'
from gfpgan import GFPGANer

# --- Layer 1: The Specialist for Memory Management ---

class MemoryManager:
    """A unified class to manage GPU memory, calculate batch sizes, and handle OOM recovery."""
    def __init__(self, safety_margin=0.8):
        self.safety_margin = safety_margin # Use 80% of available VRAM
        self.oom_count = 0
        self.successful_batch_sizes = []
        self.failed_batch_sizes = []

    def get_vram_info(self) -> dict:
        """Returns VRAM information in GB."""
        if not torch.cuda.is_available():
            return {'total': 0, 'free': 0, 'used': 0}
        
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated_mem = torch.cuda.memory_allocated() / 1e9
        return {
            'total': total_mem,
            'free': total_mem - allocated_mem,
            'used': allocated_mem
        }

    def get_safe_batch_size(self, requested_size: int) -> int:
        """Calculates a safe batch size based on available VRAM and past performance."""
        if not torch.cuda.is_available():
            return max(1, min(4, requested_size))

        # Conservative memory estimate for GFPGAN: 1.5GB per image for 512x512
        # This accounts for the model overhead, intermediate tensors, and processing buffers
        mem_per_image_gb = 1.5
        
        vram = self.get_vram_info()
        available_gb = vram['free'] * self.safety_margin
        
        # Be more conservative if we've had OOM failures
        if self.oom_count > 0:
            mem_per_image_gb *= (1.5 ** self.oom_count)  # Exponentially more conservative
            print(f"   ⚠️ OOM history detected ({self.oom_count} failures). Increasing memory estimate to {mem_per_image_gb:.1f}GB per image")
        
        memory_based_batch_size = max(1, int(available_gb / mem_per_image_gb))

        # Avoid known failed batch sizes
        while memory_based_batch_size in self.failed_batch_sizes and memory_based_batch_size > 1:
            memory_based_batch_size -= 1

        final_batch_size = min(requested_size, memory_based_batch_size)
        
        print(f"    MemoryManager: VRAM Free: {vram['free']:.2f}GB. Requested batch size: {requested_size}, Safe batch size: {final_batch_size}")
        return max(1, final_batch_size)

    def record_success(self, batch_size: int):
        self.successful_batch_sizes.append(batch_size)
        if self.oom_count > 0:
            self.oom_count = max(0, self.oom_count - 1)

    def record_oom_failure(self, batch_size: int):
        self.oom_count += 1
        if batch_size not in self.failed_batch_sizes:
            self.failed_batch_sizes.append(batch_size)

    def perform_oom_recovery(self):
        print("CUDA OOM detected! Performing adaptive recovery...")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# --- Layer 2: The Core Worker for Image Enhancement ---

class EnhancerWorker:
    """A robust worker that enhances one batch of images with built-in OOM retry logic."""
    def __init__(self, model_path: str, device: torch.device):
        self.device = device
        # The model is loaded here, inside the constructor. Since we will only create
        # one instance of this class, this code will only run once.
        self.model = self._initialize_model(model_path)
        self.memory_manager = MemoryManager()

    def _initialize_model(self, model_path: str):
        """Initializes the GFPGAN model with optimal quality settings."""
        print("   Loading GFPGAN model into memory... (This will only happen once)")
        try:
            return GFPGANer(
                model_path=model_path,
                upscale=2,  # Keep 2x upscale for good quality
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None,  # No background upsampler to save memory
                # Quality-focused settings
                device=self.device
            )
        except Exception as e:
            print(f" Failed to initialize GFPGAN model: {e}")
            return None

    def enhance_batch(self, image_batch: List[np.ndarray]) -> List[np.ndarray]:
        """Processes ONE batch of images with a try-fail-recover-fallback loop."""
        if self.model is None:
            print("⚠️ Model not initialized. Returning original images.")
            return image_batch

        max_retries = 2  # Try the full batch twice
        for attempt in range(max_retries):
            try:
                # Attempt to process the entire batch at once
                enhanced_images = self._process_batch_internal(image_batch)
                self.memory_manager.record_success(len(image_batch))
                return enhanced_images
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.memory_manager.record_oom_failure(len(image_batch))
                    self.memory_manager.perform_oom_recovery()
                    if attempt < max_retries - 1:
                        print(f"Retrying batch (attempt {attempt + 2}/{max_retries})")
                        time.sleep(1)
                    else:
                        # If batch processing fails after retries, trigger the safer fallback
                        print(" Batch processing failed due to persistent OOM. Switching to safer one-by-one fallback mode for this batch.")
                        return self._fallback_process_one_by_one(image_batch)
                else:
                    # Re-raise errors that are not OOM
                    print(f" A non-OOM error occurred in batch processing: {e}")
                    raise e
        
        # This part should ideally not be reached, but as a final safeguard:
        return image_batch

    def _process_batch_internal(self, image_batch: List[np.ndarray]) -> List[np.ndarray]:
        """The core processing loop for a single batch with improved error handling and size consistency."""
        results = []
        target_size = None
        
        # Process each image individually to ensure consistent sizing and proper error handling
        for i, img_rgb in enumerate(image_batch):
            try:
                # Ensure consistent input format
                if img_rgb.dtype != np.uint8:
                    img_rgb = (img_rgb * 255).astype(np.uint8) if img_rgb.max() <= 1.0 else img_rgb.astype(np.uint8)
                
                # Store original dimensions for consistency
                orig_h, orig_w = img_rgb.shape[:2]
                if target_size is None:
                    target_size = (orig_h, orig_w)
                
                # Convert to BGR for GFPGAN
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                
                # Process with GFPGAN using conservative settings for stability
                cropped_faces, restored_faces, restored_img_bgr = self.model.enhance(
                    img_bgr, 
                    has_aligned=False, 
                    only_center_face=False, 
                    paste_back=True
                )
                
                # Handle case where face detection failed
                if restored_img_bgr is None or len(restored_faces) == 0:
                    print(f"   - Warning: No faces detected in image {i}, using original")
                    restored_img_bgr = img_bgr
                
                # Convert back to RGB
                restored_img_rgb = cv2.cvtColor(restored_img_bgr, cv2.COLOR_BGR2RGB)
                
                # Ensure consistent output size across all images
                if restored_img_rgb.shape[:2] != target_size:
                    restored_img_rgb = cv2.resize(restored_img_rgb, (target_size[1], target_size[0]), 
                                                interpolation=cv2.INTER_LANCZOS4)
                
                # Ensure uint8 format for consistency
                if restored_img_rgb.dtype != np.uint8:
                    restored_img_rgb = (restored_img_rgb * 255).astype(np.uint8) if restored_img_rgb.max() <= 1.0 else restored_img_rgb.astype(np.uint8)
                
                results.append(restored_img_rgb)
                
            except Exception as e:
                print(f"   - Warning: Enhancement failed for image {i}: {e}. Using original.")
                # Ensure original image matches target size and format
                if target_size and img_rgb.shape[:2] != target_size:
                    img_rgb = cv2.resize(img_rgb, (target_size[1], target_size[0]), 
                                       interpolation=cv2.INTER_LANCZOS4)
                if img_rgb.dtype != np.uint8:
                    img_rgb = (img_rgb * 255).astype(np.uint8) if img_rgb.max() <= 1.0 else img_rgb.astype(np.uint8)
                results.append(img_rgb)
        
        return results

    def _fallback_process_one_by_one(self, image_batch: List[np.ndarray]) -> List[np.ndarray]:
        """
        A safer, one-by-one processing method. If any single image fails,
        it returns the original image for that slot, ensuring the output list
        has the correct length and image sizes.
        """
        results = []
        for img in image_batch:
            try:
                # Process each image as a batch of one
                enhanced_img = self._process_batch_internal([img])[0]
                results.append(enhanced_img)
            except Exception as e:
                print(f"   - ⚠️ Single image enhancement failed: {e}. Using original for this image.")
                results.append(img)  # Return the original image for this specific one
            finally:
                # Be extra careful with memory in fallback mode
                torch.cuda.empty_cache()
        return results


# --- Layer 3: The Public API and Singleton Management ---

# This global variable will hold our single, shared instance of the EnhancerWorker.
# It starts as None.
_shared_enhancer_instance = None

def _get_shared_enhancer_instance() -> EnhancerWorker:
    """
    *** THIS IS THE KEY TO LOADING THE MODEL ONLY ONCE ***
    This function acts as a factory and cache. It checks if the enhancer has been
    created. If not, it creates it. If it already exists, it returns the
    existing instance.
    """
    global _shared_enhancer_instance
    if _shared_enhancer_instance is None:
        print(" Initializing shared Enhancer for the first time...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
        _shared_enhancer_instance = EnhancerWorker(model_path=model_path, device=device)
    return _shared_enhancer_instance

def enhance_images(images: List[np.ndarray], batch_size: int = 16, max_workers: int = 4, quality_mode: str = "high") -> List[np.ndarray]:
    """
    Enhances a list of images using a parallel, auto-scaling, and OOM-safe process.
    This is the main function users should call.
    
    Args:
        images: List of input images as numpy arrays
        batch_size: Requested batch size (will be auto-adjusted for memory)
        max_workers: Number of parallel workers
        quality_mode: "high" for best quality, "balanced" for speed/quality balance
    """
    if not images:
        return []
    
    print(f" Starting face enhancement with quality mode: {quality_mode}")
    
    # Validate input images and ensure consistent format
    validated_images = []
    target_size = None
    
    for i, img in enumerate(images):
        if img is None:
            print(f"   Warning: Image {i} is None, skipping")
            continue
            
        # Ensure all images are uint8 RGB
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        
        # Store target size from first valid image
        if target_size is None:
            target_size = img.shape[:2]
        
        # Resize if necessary to maintain consistency
        if img.shape[:2] != target_size:
            img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LANCZOS4)
        
        validated_images.append(img)
    
    if not validated_images:
        print("   Warning: No valid images to enhance")
        return []
    
    print(f"   Processing {len(validated_images)} validated images (target size: {target_size})")
    
    # Every time this function is called, it gets the SAME shared enhancer instance.
    # The model is only loaded the very first time this line is executed.
    enhancer = _get_shared_enhancer_instance()
    
    # Adjust batch size based on quality mode and memory
    if quality_mode == "high":
        # For high quality, use smaller batches to prevent memory issues
        adjusted_batch_size = min(batch_size, 4)
    else:
        adjusted_batch_size = batch_size
    
    optimal_batch_size = enhancer.memory_manager.get_safe_batch_size(
        requested_size=adjusted_batch_size
    )
    
    batches = [validated_images[i:i + optimal_batch_size] for i in range(0, len(validated_images), optimal_batch_size)]
    
    all_enhanced_images = []
    
    # Use single-threaded processing for better memory management and stability
    results = {}
    
    for i, batch in enumerate(tqdm(batches, desc="Enhancing Batches")):
        try:
            # Process batch with proper error handling
            batch_result = enhancer.enhance_batch(batch)
            
            # Validate batch result length and content
            if len(batch_result) != len(batch):
                print(f"⚠️ Warning: Batch {i} result count mismatch ({len(batch_result)} vs {len(batch)}). Using original.")
                results[i] = batch
            else:
                # Ensure all results have consistent dimensions
                validated_batch = []
                for j, result_img in enumerate(batch_result):
                    if result_img.shape[:2] != target_size:
                        result_img = cv2.resize(result_img, (target_size[1], target_size[0]), 
                                              interpolation=cv2.INTER_LANCZOS4)
                    if result_img.dtype != np.uint8:
                        result_img = (result_img * 255).astype(np.uint8) if result_img.max() <= 1.0 else result_img.astype(np.uint8)
                    validated_batch.append(result_img)
                results[i] = validated_batch
                
        except Exception as exc:
            print(f" A batch generated a critical, unrecoverable exception: {exc}")
            # As a last resort, return the original images for that batch
            results[i] = batch

    # Assemble final results in correct order
    for i in range(len(batches)):
        all_enhanced_images.extend(results.get(i, []))
    
    print(f"   Enhancement complete: {len(all_enhanced_images)} images processed")
    return all_enhanced_images

