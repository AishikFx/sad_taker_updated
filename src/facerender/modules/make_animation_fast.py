"""
Optimized Face Renderer with batched processing for significant speedup
This addresses the Face Renderer bottleneck (4.64s per iteration x 7 = ~32s total)
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from scipy.spatial import ConvexHull

def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).type_as(pred).to(device)
    pred = F.softmax(pred)
    degree = torch.sum(pred*idx_tensor, 1) * 3 - 99
    return degree

def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    pitch_mat = torch.cat([torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch), 
                          torch.zeros_like(pitch), torch.cos(pitch), -torch.sin(pitch),
                          torch.zeros_like(pitch), torch.sin(pitch), torch.cos(pitch)], dim=1)
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat([torch.cos(yaw), torch.zeros_like(yaw), torch.sin(yaw), 
                           torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw),
                           -torch.sin(yaw), torch.zeros_like(yaw), torch.cos(yaw)], dim=1)
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    roll_mat = torch.cat([torch.cos(roll), -torch.sin(roll), torch.zeros_like(roll),  
                         torch.sin(roll), torch.cos(roll), torch.zeros_like(roll),
                         torch.zeros_like(roll), torch.zeros_like(roll), torch.ones_like(roll)], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    rot_mat = torch.einsum('bij,bjk,bkm->bim', pitch_mat, yaw_mat, roll_mat)
    return rot_mat

def keypoint_transformation(kp_canonical, he, wo_exp=False):
    kp = kp_canonical['value']    # (bs, k, 3) 
    yaw, pitch, roll= he['yaw'], he['pitch'], he['roll']      
    yaw = headpose_pred_to_degree(yaw) 
    pitch = headpose_pred_to_degree(pitch)
    roll = headpose_pred_to_degree(roll)

    if 'yaw_in' in he:
        yaw = he['yaw_in']
    if 'pitch_in' in he:
        pitch = he['pitch_in']
    if 'roll_in' in he:
        roll = he['roll_in']

    rot_mat = get_rotation_matrix(yaw, pitch, roll)    # (bs, 3, 3)

    t, exp = he['t'], he['exp']
    if wo_exp:
        exp =  exp*0  
    
    # keypoint rotation
    kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, kp)

    # keypoint translation
    t[:, 0] = t[:, 0]*0
    t[:, 2] = t[:, 2]*0
    t = t.unsqueeze(1).repeat(1, kp.shape[1], 1)
    kp_t = kp_rotated + t

    # add expression deviation 
    exp = exp.view(exp.shape[0], -1, 3)
    kp_transformed = kp_t + exp

    return {'value': kp_transformed}


class OptimizedFaceRenderer:
    """Optimized Face Renderer with multiple acceleration strategies"""
    
    def __init__(self, optimization_level="medium"):
        self.optimization_level = optimization_level
        
        # Set batch sizes based on optimization level
        if optimization_level == "extreme":
            self.batch_size = 16
            self.use_mixed_precision = True
            self.aggressive_caching = True
        elif optimization_level == "high":
            self.batch_size = 12
            self.use_mixed_precision = True
            self.aggressive_caching = False
        elif optimization_level == "medium":
            self.batch_size = 8
            self.use_mixed_precision = False
            self.aggressive_caching = False
        else:  # low
            self.batch_size = 4
            self.use_mixed_precision = False
            self.aggressive_caching = False
    
    def auto_detect_batch_size(self):
        """Auto-detect optimal batch size based on GPU memory"""
        if not torch.cuda.is_available():
            return 2
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        if gpu_memory >= 24:
            return min(16, self.batch_size)
        elif gpu_memory >= 16:
            return min(12, self.batch_size)
        elif gpu_memory >= 12:
            return min(8, self.batch_size)
        elif gpu_memory >= 8:
            return min(6, self.batch_size)
        elif gpu_memory >= 6:
            return min(4, self.batch_size)
        else:
            return 2
    
    def make_animation_optimized(self, source_image, source_semantics, target_semantics,
                                generator, kp_detector, he_estimator, mapping, 
                                yaw_c_seq=None, pitch_c_seq=None, roll_c_seq=None,
                                use_exp=True):
        """
        Optimized version that processes multiple frames in parallel
        Expected speedup: 3-5x faster than original sequential processing
        """
        
        # Auto-detect optimal batch size
        optimal_batch_size = self.auto_detect_batch_size()
        batch_size = min(optimal_batch_size, self.batch_size)
        
        with torch.no_grad():
            # Enable mixed precision if configured
            if self.use_mixed_precision and torch.cuda.is_available():
                autocast = torch.cuda.amp.autocast
            else:
                autocast = torch.nullcontext
            
            with autocast():
                # Pre-compute source keypoints once (major optimization)
                kp_canonical = kp_detector(source_image)
                he_source = mapping(source_semantics)
                kp_source = keypoint_transformation(kp_canonical, he_source)
                
                total_frames = target_semantics.shape[1]
                predictions = []
                
                # Process frames in batches instead of one by one
                desc = f'Face Renderer (Batch={batch_size})'
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
                    kp_driving_batch = keypoint_transformation(kp_canonical_batch, he_driving_batch)
                    
                    # Batch generation - THIS IS THE KEY SPEEDUP
                    source_image_batch = source_image.repeat(current_batch_size, 1, 1, 1)
                    kp_source_batch = {k: v.repeat(current_batch_size, 1, 1) for k, v in kp_source.items()}
                    
                    # Generate all frames in this batch at once
                    out_batch = generator(source_image_batch, kp_source=kp_source_batch, kp_driving=kp_driving_batch)
                    
                    # Reshape batch predictions back to sequence format
                    batch_predictions = out_batch['prediction'].reshape(1, current_batch_size, *out_batch['prediction'].shape[1:])
                    predictions.append(batch_predictions)
                    
                    # Periodic GPU memory cleanup
                    if torch.cuda.is_available() and len(predictions) % 4 == 0:
                        torch.cuda.empty_cache()
                
                # Concatenate all predictions
                predictions_ts = torch.cat(predictions, dim=1)
        
        return predictions_ts
    
    def make_animation_streaming(self, source_image, source_semantics, target_semantics,
                               generator, kp_detector, he_estimator, mapping, 
                               yaw_c_seq=None, pitch_c_seq=None, roll_c_seq=None,
                               use_exp=True, max_memory_frames=50):
        """
        Memory-efficient streaming version for very long videos
        Processes in chunks to avoid OOM on long sequences
        """
        
        total_frames = target_semantics.shape[1]
        if total_frames <= max_memory_frames:
            # Use regular optimized version for short videos
            return self.make_animation_optimized(
                source_image, source_semantics, target_semantics,
                generator, kp_detector, he_estimator, mapping,
                yaw_c_seq, pitch_c_seq, roll_c_seq, use_exp
            )
        
        # Stream processing for long videos
        predictions_chunks = []
        
        for chunk_start in tqdm(range(0, total_frames, max_memory_frames), 'Streaming Face Render'):
            chunk_end = min(chunk_start + max_memory_frames, total_frames)
            
            # Extract chunk
            chunk_target_semantics = target_semantics[:, chunk_start:chunk_end]
            chunk_yaw = yaw_c_seq[:, chunk_start:chunk_end] if yaw_c_seq is not None else None
            chunk_pitch = pitch_c_seq[:, chunk_start:chunk_end] if pitch_c_seq is not None else None
            chunk_roll = roll_c_seq[:, chunk_start:chunk_end] if roll_c_seq is not None else None
            
            # Process chunk
            chunk_predictions = self.make_animation_optimized(
                source_image, source_semantics, chunk_target_semantics,
                generator, kp_detector, he_estimator, mapping,
                chunk_yaw, chunk_pitch, chunk_roll, use_exp
            )
            
            predictions_chunks.append(chunk_predictions)
            
            # Cleanup between chunks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return torch.cat(predictions_chunks, dim=1)


def make_animation_fast(source_image, source_semantics, target_semantics,
                       generator, kp_detector, he_estimator, mapping, 
                       yaw_c_seq=None, pitch_c_seq=None, roll_c_seq=None,
                       use_exp=True, optimization_level="medium"):
    """
    Fast replacement for the original make_animation function
    Expected speedup: 3-5x faster
    """
    
    renderer = OptimizedFaceRenderer(optimization_level=optimization_level)
    return renderer.make_animation_optimized(
        source_image, source_semantics, target_semantics,
        generator, kp_detector, he_estimator, mapping,
        yaw_c_seq, pitch_c_seq, roll_c_seq, use_exp
    )


# Backwards compatibility wrapper
def make_animation(*args, **kwargs):
    """Backwards compatible wrapper that uses optimized version"""
    # Extract optimization level from kwargs if provided
    optimization_level = kwargs.pop('optimization_level', 'medium')
    return make_animation_fast(*args, optimization_level=optimization_level, **kwargs)