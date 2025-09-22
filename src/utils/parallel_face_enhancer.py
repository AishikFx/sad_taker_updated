import os
import time
import uuid
from typing import List, Dict
import cv2
import numpy as np
import torch
from tqdm import tqdm

# Import the new VRAM queue manager
from src.utils.vram_queue_manager import (
    ProcessTask, get_global_processor, ParallelVRAMProcessor
)

# Import the existing face enhancer for GFPGAN model
try:
    from gfpgan import GFPGANer
except ImportError:
    print("Warning: GFPGAN not installed. Face enhancement will be disabled.")
    GFPGANer = None


class ParallelFaceEnhancer:
    """Face enhancer using parallel VRAM-managed processing"""
    
    def __init__(self, model_path: str = None, upscale: int = 2):
        self.model_path = model_path or 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
        self.upscale = upscale
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # VRAM estimation for different image sizes
        self.vram_estimates = {
            256: 1.2,   # 1.2GB for 256x256
            512: 2.0,   # 2.0GB for 512x512  
            1024: 3.5,  # 3.5GB for 1024x1024
        }
        
        self.processor = get_global_processor()
        print(f"Parallel Face Enhancer initialized on {self.device}")
    
    def _get_model(self):
        """Lazy load the GFPGAN model"""
        if self.model is None and GFPGANer is not None:
            print("Loading GFPGAN model...")
            try:
                self.model = GFPGANer(
                    model_path=self.model_path,
                    upscale=self.upscale,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=None,
                    device=self.device
                )
                print("GFPGAN model loaded successfully")
            except Exception as e:
                print(f"Failed to load GFPGAN model: {e}")
                self.model = None
        return self.model
    
    def _estimate_vram_for_image(self, image: np.ndarray) -> float:
        """Estimate VRAM needed for processing this image"""
        height, width = image.shape[:2]
        max_dim = max(height, width)
        
        # Find the closest size estimate
        for size, vram in sorted(self.vram_estimates.items()):
            if max_dim <= size:
                return vram
        
        # For very large images, extrapolate
        return self.vram_estimates[1024] * (max_dim / 1024) ** 2
    
    def _enhance_single_image(self, image_data: Dict) -> np.ndarray:
        """Process a single image (used as callback)"""
        image = image_data['image']
        image_id = image_data['id']
        target_size = image_data.get('target_size')
        
        try:
            model = self._get_model()
            if model is None:
                print(f"Warning: GFPGAN model not available for image {image_id}, returning original")
                return self._ensure_consistent_format(image, target_size)
            
            # Ensure proper format
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
            
            # Convert to BGR for GFPGAN
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Process with GFPGAN
            cropped_faces, restored_faces, restored_img_bgr = model.enhance(
                img_bgr, 
                has_aligned=False, 
                only_center_face=False, 
                paste_back=True
            )
            
            # Handle case where face detection failed
            if restored_img_bgr is None or len(restored_faces) == 0:
                print(f"Warning: No faces detected in image {image_id}, using original")
                restored_img_bgr = img_bgr
            
            # Convert back to RGB
            restored_img_rgb = cv2.cvtColor(restored_img_bgr, cv2.COLOR_BGR2RGB)
            
            # Ensure consistent output format
            return self._ensure_consistent_format(restored_img_rgb, target_size)
            
        except Exception as e:
            print(f"Error enhancing image {image_id}: {e}")
            return self._ensure_consistent_format(image, target_size)
    
    def _ensure_consistent_format(self, image: np.ndarray, target_size: tuple = None) -> np.ndarray:
        """Ensure image has consistent format and size"""
        # Ensure uint8 format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        # Resize if target size specified
        if target_size and image.shape[:2] != target_size:
            image = cv2.resize(image, (target_size[1], target_size[0]), 
                             interpolation=cv2.INTER_LANCZOS4)
        
        return image
    
    def enhance_images_parallel(self, images: List[np.ndarray], quality_mode: str = "high") -> List[np.ndarray]:
        """
        Enhance images using parallel VRAM-managed processing
        
        Args:
            images: List of input images
            quality_mode: "high" for best quality, "balanced" for speed
        """
        if not images:
            return []
        
        print(f"Starting parallel face enhancement of {len(images)} images (quality: {quality_mode})")
        
        # Determine target size from first valid image
        target_size = None
        for img in images:
            if img is not None:
                target_size = img.shape[:2]
                break
        
        if target_size is None:
            print("No valid images found")
            return []
        
        print(f"Target size: {target_size}")
        
        # Create tasks for each image
        tasks = []
        for i, image in enumerate(images):
            if image is None:
                print(f"Warning: Image {i} is None, skipping")
                continue
            
            # Estimate VRAM requirement
            vram_required = self._estimate_vram_for_image(image)
            
            # Adjust VRAM estimate based on quality mode
            if quality_mode == "high":
                vram_required *= 1.2  # 20% safety margin for high quality
            
            # Create task
            task_data = {
                'image': image,
                'id': i,
                'target_size': target_size
            }
            
            task = ProcessTask(
                task_id=f"enhance_{i}_{uuid.uuid4().hex[:8]}",
                data=task_data,
                vram_required_gb=vram_required,
                priority=0,  # FIFO processing
                callback=self._enhance_single_image
            )
            
            tasks.append((i, task))
        
        print(f"Created {len(tasks)} enhancement tasks")
        
        # Submit all tasks to the parallel processor
        task_results = {}
        for original_index, task in tasks:
            self.processor.submit_task(task)
            task_results[task.task_id] = original_index
        
        # Wait for all tasks to complete
        print("Waiting for all enhancement tasks to complete...")
        
        # Monitor progress
        start_time = time.time()
        last_status_time = start_time
        
        while not self.processor.wait_for_completion(timeout=1.0):
            current_time = time.time()
            
            # Print status every 5 seconds
            if current_time - last_status_time >= 5.0:
                status = self.processor.get_status()
                print(f"Enhancement progress: {status['active_tasks_count']} active, "
                      f"{status['pending_queue_size']} pending, "
                      f"{status['completed_count']} completed, "
                      f"VRAM allocated: {status['total_vram_allocated']:.1f}GB")
                last_status_time = current_time
            
            # Timeout after 5 minutes
            if current_time - start_time > 300:
                print("Warning: Enhancement timeout after 5 minutes")
                break
        
        # Collect results in original order
        enhanced_images = [None] * len(images)
        
        for task in self.processor.completed_tasks:
            if task.task_id in task_results:
                original_index = task_results[task.task_id]
                if task.result is not None:
                    enhanced_images[original_index] = task.result
                else:
                    print(f"Warning: Task {task.task_id} completed but no result")
                    # Use original image as fallback
                    enhanced_images[original_index] = self._ensure_consistent_format(
                        images[original_index], target_size
                    )
        
        # Handle failed tasks
        for task in self.processor.failed_tasks:
            if task.task_id in task_results:
                original_index = task_results[task.task_id]
                print(f"Warning: Enhancement failed for image {original_index}, using original")
                enhanced_images[original_index] = self._ensure_consistent_format(
                    images[original_index], target_size
                )
        
        # Fill any remaining None values with original images
        for i, enhanced in enumerate(enhanced_images):
            if enhanced is None and i < len(images) and images[i] is not None:
                print(f"Warning: No result for image {i}, using original")
                enhanced_images[i] = self._ensure_consistent_format(images[i], target_size)
        
        # Filter out None values and ensure we have results
        valid_enhanced = [img for img in enhanced_images if img is not None]
        
        final_status = self.processor.get_status()
        print(f"Enhancement completed: {len(valid_enhanced)}/{len(images)} images processed successfully")
        print(f"Final stats: {final_status['stats']}")
        
        return valid_enhanced


# Convenience function for backward compatibility
def enhance_images(images: List[np.ndarray], batch_size: int = 16, 
                  max_workers: int = 4, quality_mode: str = "high") -> List[np.ndarray]:
    """
    Enhanced image enhancement using parallel VRAM management
    
    Note: batch_size and max_workers are ignored in favor of dynamic VRAM management
    """
    enhancer = ParallelFaceEnhancer()
    return enhancer.enhance_images_parallel(images, quality_mode=quality_mode)