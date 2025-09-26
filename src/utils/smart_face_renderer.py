"""
Smart Face Renderer with High-Quality Natural Animation
Focused on preserving animation quality and natural expressions
"""

import os
import time
import gc
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from contextlib import nullcontext

# Import the basic animation functions we need
from src.facerender.modules.make_animation import get_rotation_matrix, keypoint_transformation

# --- Layer 1: Quality-Focused Memory Manager for Face Renderer ---

class QualityFaceRenderMemoryManager:
    """Quality-focused memory manager for face rendering with stable performance."""
    
    def __init__(self, safety_margin=0.75):  # Conservative safety margin for stability
        self.safety_margin = safety_margin  
        self.oom_count = 0
        self.successful_batch_sizes = []
        self.failed_batch_sizes = []
        self.last_successful_batch_size = None

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

    def estimate_face_render_memory_per_frame(self, img_size: int = 256) -> float:
        """Conservative memory estimation for stable face rendering."""
        # Conservative memory estimation for stability
        base_memory_gb = 0.5  # Conservative base estimate
        resolution_factor = (img_size / 256) ** 2  # Linear scaling for safety
        memory_per_frame = base_memory_gb * resolution_factor
        overhead_factor = 1.3  # Conservative overhead
        return memory_per_frame * overhead_factor
        return memory_per_frame * overhead_factor

    def get_safe_batch_size(self, requested_size: int, img_size: int = 256) -> int:
        """Calculate safe batch size for face rendering based on available VRAM."""
        if not torch.cuda.is_available():
            return max(1, min(8, requested_size))  # CPU fallback
        
        vram = self.get_vram_info()
        available_gb = vram['free'] * self.safety_margin
        
        memory_per_frame = self.estimate_face_render_memory_per_frame(img_size)
        memory_based_batch_size = max(1, int(available_gb / memory_per_frame))
        
        # Apply OOM penalty if we've had failures
        if self.oom_count > 0:
            penalty_factor = 2 ** self.oom_count
            memory_based_batch_size = max(1, memory_based_batch_size // penalty_factor)
        
        # Avoid known failed batch sizes
        while memory_based_batch_size in self.failed_batch_sizes and memory_based_batch_size > 1:
            memory_based_batch_size -= 1
        
        # Use last successful batch size as upper bound if available
        if self.last_successful_batch_size:
            memory_based_batch_size = min(memory_based_batch_size, self.last_successful_batch_size + 2)
        
        final_batch_size = min(requested_size, memory_based_batch_size)
        
        return max(1, final_batch_size)

    def record_success(self, batch_size: int):
        """Record a successful batch processing."""
        self.successful_batch_sizes.append(batch_size)
        self.last_successful_batch_size = batch_size
        if self.oom_count > 0:
            self.oom_count = max(0, self.oom_count - 1)

    def record_oom_failure(self, batch_size: int):
        """Record an OOM failure."""
        self.oom_count += 1
        if batch_size not in self.failed_batch_sizes:
            self.failed_batch_sizes.append(batch_size)

    def perform_oom_recovery(self):
        """Perform memory cleanup after OOM."""
        print("Face Renderer CUDA OOM detected! Performing recovery...")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# --- Layer 2: Smart Face Renderer Worker ---

class SmartFaceRenderWorker:
    """A smart face renderer that supports natural animation and handles OOM gracefully."""
    
    def __init__(self, optimization_level: str = "medium", natural_animation: bool = True):
        self.optimization_level = optimization_level
        self.natural_animation = natural_animation
        self.memory_manager = QualityFaceRenderMemoryManager()
        
        # Quality-focused settings
        self.use_mixed_precision = False  # Disable for quality
        self.aggressive_batching = False  # Disable for stability
        
        # Performance tracking
        self.total_frames_processed = 0
        self.total_processing_time = 0.0
        
        print(f" SmartFaceRenderWorker initialized:")
        print(f"    Optimization: {optimization_level}")
        print(f"    Natural Animation: {'Enabled (raw keypoints for realism)' if natural_animation else 'Disabled (optimized keypoints)'}")

    def render_animation_smart(self, 
                              source_image: torch.Tensor, 
                              source_semantics: torch.Tensor, 
                              target_semantics: torch.Tensor,
                              generator, 
                              kp_detector, 
                              he_estimator, 
                              mapping, 
                              yaw_c_seq: Optional[torch.Tensor] = None,
                              pitch_c_seq: Optional[torch.Tensor] = None,
                              roll_c_seq: Optional[torch.Tensor] = None,
                              use_exp: bool = True,
                              requested_batch_size: int = 8) -> torch.Tensor:
        """Smart animation rendering with natural animation support."""
        
        start_time = time.time()
        frame_count = target_semantics.shape[1]
        
        img_size = source_image.shape[-1]
        optimal_batch_size = self.memory_manager.get_safe_batch_size(requested_batch_size, img_size)
        
        print(f" Smart Face Renderer: Processing {frame_count} frames")
        if self.natural_animation:
            print(f"    Using natural animation mode for maximum realism")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = self._render_with_batch_size(
                    source_image, source_semantics, target_semantics,
                    generator, kp_detector, he_estimator, mapping,
                    yaw_c_seq, pitch_c_seq, roll_c_seq, use_exp,
                    optimal_batch_size
                )
                
                # Record success and performance metrics
                self.memory_manager.record_success(optimal_batch_size)
                processing_time = time.time() - start_time
                self.total_frames_processed += frame_count
                self.total_processing_time += processing_time
                
                fps = frame_count / processing_time
                print(f" Face Rendering Complete! {fps:.2f} FPS ({processing_time:.2f}s)")
                
                return result
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.memory_manager.record_oom_failure(optimal_batch_size)
                    self.memory_manager.perform_oom_recovery()
                    optimal_batch_size = max(1, optimal_batch_size // 2)
                    
                    if attempt < max_retries - 1:
                        print(f"Face Renderer OOM retry {attempt + 2}/{max_retries} with batch size {optimal_batch_size}")
                        time.sleep(1)
                    else:
                        return self._render_with_batch_size(
                            source_image, source_semantics, target_semantics,
                            generator, kp_detector, he_estimator, mapping,
                            yaw_c_seq, pitch_c_seq, roll_c_seq, use_exp, 1
                        )
                else:
                    raise e

    def _render_with_batch_size(self,
                               source_image: torch.Tensor, 
                               source_semantics: torch.Tensor, 
                               target_semantics: torch.Tensor,
                               generator, 
                               kp_detector, 
                               he_estimator, 
                               mapping,
                               yaw_c_seq: Optional[torch.Tensor],
                               pitch_c_seq: Optional[torch.Tensor],
                               roll_c_seq: Optional[torch.Tensor],
                               use_exp: bool,
                               batch_size: int) -> torch.Tensor:
        """Core rendering function with specified batch size."""
        
        # Dynamic memory management based on available VRAM
        if torch.cuda.is_available():
            total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            available_memory_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3
            
            # Calculate memory ratio for dynamic thresholds
            memory_ratio = total_memory_gb / 24.0
            available_ratio = available_memory_gb / 24.0
            
            # Adaptive memory cleanup based on available memory ratio
            if available_ratio < 0.17:  # Less than 17% of 24GB (4GB)
                # Aggressive cleanup for low memory
                import gc
                for _ in range(3):
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            elif available_ratio < 0.33:  # Less than 33% of 24GB (8GB)
                # Moderate cleanup for medium memory
                torch.cuda.empty_cache()
        
        with torch.no_grad():
            # Dynamic mixed precision based on memory availability
            use_autocast = False
            if torch.cuda.is_available():
                total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                available_memory_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3
                memory_ratio = total_memory_gb / 24.0
                available_ratio = available_memory_gb / 24.0
                # Only use mixed precision if we have sufficient memory (saves memory overhead)
                use_autocast = self.use_mixed_precision and available_ratio >= 0.25  # 25% of 24GB (6GB)
            
            if use_autocast:
                try:
                    autocast = torch.cuda.amp.autocast
                except AttributeError:
                    autocast = nullcontext
            else:
                autocast = nullcontext
            
            with autocast():
                # Pre-compute source keypoints once with immediate cleanup
                kp_canonical = kp_detector(source_image)
                
                # Immediate cleanup after keypoint detection if low memory
                if torch.cuda.is_available():
                    total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    available_memory_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3
                    available_ratio = available_memory_gb / 24.0
                    if available_ratio < 0.25:  # Less than 25% of 24GB (6GB)
                        torch.cuda.empty_cache()
                
                he_source = mapping(source_semantics)
                kp_source = keypoint_transformation(kp_canonical, he_source, wo_exp=False)
                
                total_frames = target_semantics.shape[1]
                predictions = []
                
                # Process frames in batches for GPU efficiency
                desc = f'Smart Face Renderer (Batch={batch_size}, Natural Animation)' if self.natural_animation else f'Smart Face Renderer (Batch={batch_size}, Optimized)'
                for start_idx in tqdm(range(0, total_frames, batch_size), desc):
                    end_idx = min(start_idx + batch_size, total_frames)
                    current_batch_size = end_idx - start_idx
                    
                    # Prepare batch data efficiently
                    batch_target_semantics = target_semantics[:, start_idx:end_idx]
                    batch_target_semantics = batch_target_semantics.reshape(-1, batch_target_semantics.shape[-1])
                    
                    # Batch process driving parameters
                    he_driving_batch = mapping(batch_target_semantics)
                    
                    # Handle pose sequences in batch
                    if yaw_c_seq is not None:
                        he_driving_batch['yaw_in'] = yaw_c_seq[:, start_idx:end_idx].reshape(-1)
                    if pitch_c_seq is not None:
                        he_driving_batch['pitch_in'] = pitch_c_seq[:, start_idx:end_idx].reshape(-1)
                    if roll_c_seq is not None:
                        he_driving_batch['roll_in'] = roll_c_seq[:, start_idx:end_idx].reshape(-1)
                    
                    # Batch keypoint transformation
                    kp_canonical_batch = {k: v.repeat(current_batch_size, 1, 1) for k, v in kp_canonical.items()}
                    kp_driving_batch = keypoint_transformation(kp_canonical_batch, he_driving_batch, wo_exp=False)
                    
                    # Choose animation approach based on natural_animation setting
                    if self.natural_animation:
                        # Natural mode: Use raw keypoints like original SadTalker for maximum realism
                        kp_norm_batch = kp_driving_batch
                    else:
                        # Optimized mode: Use keypoint normalization (may reduce some micro-expressions)
                        # For now, we'll still use raw keypoints for best quality
                        kp_norm_batch = kp_driving_batch
                    
                    # Batch generation - THIS IS THE KEY GPU EFFICIENCY IMPROVEMENT
                    source_image_batch = source_image.repeat(current_batch_size, 1, 1, 1)
                    kp_source_batch = {k: v.repeat(current_batch_size, 1, 1) for k, v in kp_source.items()}
                    
                    # Generate all frames in this batch at once
                    out_batch = generator(source_image_batch, kp_source=kp_source_batch, kp_driving=kp_norm_batch)
                    
                    # Reshape batch predictions back to sequence format
                    batch_predictions = out_batch['prediction'].reshape(1, current_batch_size, *out_batch['prediction'].shape[1:])
                    predictions.append(batch_predictions)
                    
                    # Dynamic GPU memory cleanup based on available memory
                    if torch.cuda.is_available() and len(predictions) % 4 == 0:
                        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        available_memory_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3
                        available_ratio = available_memory_gb / 24.0
                        
                        # More frequent cleanup for low memory systems
                        cleanup_frequency = 2 if available_ratio < 0.25 else 4  # 25% of 24GB (6GB)
                        if len(predictions) % cleanup_frequency == 0:
                            torch.cuda.empty_cache()
                            if available_ratio < 0.17:  # Less than 17% of 24GB (4GB)
                                import gc
                                gc.collect()
                
                # Concatenate all batch predictions
                predictions_ts = torch.cat(predictions, dim=1)
                
                # Final cleanup for low memory systems
                if torch.cuda.is_available():
                    total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    available_memory_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3
                    available_ratio = available_memory_gb / 24.0
                    if available_ratio < 0.25:  # Less than 25% of 24GB (6GB)
                        torch.cuda.empty_cache()
        
        return predictions_ts


# --- Layer 3: Public API ---

# Global singleton instance
_smart_face_renderer_instance = None

def get_smart_face_renderer(optimization_level: str = "medium", natural_animation: bool = True) -> SmartFaceRenderWorker:
    """Get or create the global smart face renderer instance."""
    global _smart_face_renderer_instance
    recreate_needed = (
        _smart_face_renderer_instance is None or 
        _smart_face_renderer_instance.optimization_level != optimization_level or
        _smart_face_renderer_instance.natural_animation != natural_animation
    )
    
    if recreate_needed:
        _smart_face_renderer_instance = SmartFaceRenderWorker(optimization_level, natural_animation)
    return _smart_face_renderer_instance

def render_animation_smart(source_image: torch.Tensor, 
                          source_semantics: torch.Tensor, 
                          target_semantics: torch.Tensor,
                          generator, 
                          kp_detector, 
                          he_estimator, 
                          mapping, 
                          yaw_c_seq: Optional[torch.Tensor] = None,
                          pitch_c_seq: Optional[torch.Tensor] = None,
                          roll_c_seq: Optional[torch.Tensor] = None,
                          use_exp: bool = True,
                          optimization_level: str = "medium",
                          natural_animation: bool = True,
                          batch_size: int = 8) -> torch.Tensor:
    """
    Main public function for smart face rendering with natural animation support.
    
    Args:
        natural_animation: If True, uses raw keypoints for maximum realism (RECOMMENDED)
                          If False, may apply optimizations that reduce some micro-expressions
    
    This function provides:
    - Natural eye blinking and micro-expressions
    - Dynamic VRAM management
    - OOM error recovery
    - Performance optimizations while maintaining quality
    """
    
    renderer = get_smart_face_renderer(optimization_level, natural_animation)
    
    return renderer.render_animation_smart(
        source_image, source_semantics, target_semantics,
        generator, kp_detector, he_estimator, mapping,
        yaw_c_seq, pitch_c_seq, roll_c_seq, use_exp,
        batch_size
    )