# src/utils/fast_paste_pic.py
import cv2
import os
import numpy as np
from tqdm import tqdm
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from numba import jit, cuda
import threading
from queue import Queue

from src.utils.videoio import save_video_with_watermark


@jit(nopython=True)
def fast_alpha_blend(src, dst, alpha, x1, y1, x2, y2, src_x1, src_y1, src_x2, src_y2):
    """Numba-optimized alpha blending"""
    for i in range(y1, y2):
        for j in range(x1, x2):
            src_i = i - y1 + src_y1
            src_j = j - x1 + src_x1
            if src_i < src_y2 and src_j < src_x2:
                alpha_val = alpha[src_i, src_j] / 255.0
                for c in range(3):  # RGB channels
                    dst[i, j, c] = int(src[src_i, src_j, c] * alpha_val + dst[i, j, c] * (1 - alpha_val))


class FastSeamlessClone:
    """Fast alternatives to cv2.seamlessClone"""
    
    @staticmethod
    def simple_blend(src, dst, mask, center):
        """Fast simple blending without seamless clone"""
        alpha = mask.astype(np.float32) / 255.0
        h, w = src.shape[:2]
        cx, cy = center
        
        x1 = max(0, cx - w//2)
        y1 = max(0, cy - h//2)
        x2 = min(dst.shape[1], x1 + w)
        y2 = min(dst.shape[0], y1 + h)
        
        src_x1 = max(0, w//2 - cx)
        src_y1 = max(0, h//2 - cy)
        src_x2 = src_x1 + (x2 - x1)
        src_y2 = src_y1 + (y2 - y1)
        
        if x2 > x1 and y2 > y1 and src_x2 > src_x1 and src_y2 > src_y1:
            dst_region = dst[y1:y2, x1:x2]
            src_region = src[src_y1:src_y2, src_x1:src_x2]
            alpha_region = alpha[src_y1:src_y2, src_x1:src_x2]
            
            # Vectorized blending
            blended = src_region * alpha_region[..., np.newaxis] + dst_region * (1 - alpha_region[..., np.newaxis])
            dst[y1:y2, x1:x2] = blended.astype(np.uint8)
        
        return dst
    
    @staticmethod
    def gaussian_blend(src, dst, center, blend_radius=50):
        """Gaussian blending for smoother results"""
        h, w = src.shape[:2]
        cx, cy = center
        
        # Create Gaussian mask
        y, x = np.ogrid[:h, :w]
        mask = np.exp(-((x - w//2)**2 + (y - h//2)**2) / (2 * blend_radius**2))
        mask = (mask * 255).astype(np.uint8)
        
        return FastSeamlessClone.simple_blend(src, dst, mask, center)
    
    @staticmethod
    def feather_blend(src, dst, center, feather_size=20):
        """Fast feathered blending"""
        h, w = src.shape[:2]
        
        # Create feathered mask
        mask = np.ones((h, w), dtype=np.uint8) * 255
        mask[:feather_size, :] = 0
        mask[-feather_size:, :] = 0
        mask[:, :feather_size] = 0
        mask[:, -feather_size:] = 0
        
        # Apply distance transform for smooth feathering
        mask = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        mask = np.clip(mask / feather_size * 255, 0, 255).astype(np.uint8)
        
        return FastSeamlessClone.simple_blend(src, dst, mask, center)


class OptimizedPastePic:
    """Optimized paste_pic with multiple speed/quality modes"""
    
    def __init__(self, optimization_level="medium", blend_method="simple"):
        """
        optimization_level: "low", "medium", "high", "extreme"
        blend_method: "seamless", "simple", "gaussian", "feather"
        """
        self.optimization_level = optimization_level
        self.blend_method = blend_method
        
        # Determine processing parameters
        if optimization_level == "extreme":
            self.use_parallel = True
            self.batch_size = 16
            self.num_workers = min(8, multiprocessing.cpu_count())
            if blend_method == "seamless":
                self.blend_method = "simple"  # Force simple blending for speed
        elif optimization_level == "high":
            self.use_parallel = True
            self.batch_size = 8
            self.num_workers = min(4, multiprocessing.cpu_count())
        elif optimization_level == "medium":
            self.use_parallel = True
            self.batch_size = 4
            self.num_workers = 2
        else:
            self.use_parallel = False
            self.batch_size = 1
            self.num_workers = 1
    
    def paste_video(self, video_path, pic_path, crop_info, new_audio_path, full_video_path, extended_crop=False):
        """Main paste function with optimizations"""
        
        if not os.path.isfile(pic_path):
            raise ValueError('pic_path must be a valid path to video/image file')
        
        # Load background image/frame
        full_img = self._load_background(pic_path)
        frame_h, frame_w = full_img.shape[:2]
        
        # Load video frames
        crop_frames = self._load_video_frames(video_path)
        
        if len(crop_info) != 3:
            print("you didn't crop the image")
            return
        
        # Calculate crop coordinates
        r_w, r_h = crop_info[0]
        clx, cly, crx, cry = crop_info[1] 
        lx, ly, rx, ry = crop_info[2]
        lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
        
        if extended_crop:
            oy1, oy2, ox1, ox2 = cly, cry, clx, crx
        else:
            oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx
        
        # Process frames
        if self.use_parallel and len(crop_frames) > self.batch_size:
            processed_frames = self._parallel_process_frames(
                crop_frames, full_img, ox1, oy1, ox2, oy2, frame_w, frame_h
            )
        else:
            processed_frames = self._sequential_process_frames(
                crop_frames, full_img, ox1, oy1, ox2, oy2, frame_w, frame_h
            )
        
        # Save video
        self._save_processed_video(processed_frames, video_path, new_audio_path, full_video_path)
    
    def _load_background(self, pic_path):
        """Load background image or first frame of video"""
        if pic_path.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
            return cv2.imread(pic_path)
        else:
            video_stream = cv2.VideoCapture(pic_path)
            ret, frame = video_stream.read()
            video_stream.release()
            return frame if ret else None
    
    def _load_video_frames(self, video_path):
        """Load all video frames"""
        video_stream = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = video_stream.read()
            if not ret:
                break
            frames.append(frame)
        video_stream.release()
        return frames
    
    def _parallel_process_frames(self, crop_frames, full_img, ox1, oy1, ox2, oy2, frame_w, frame_h):
        """Process frames in parallel"""
        processed_frames = [None] * len(crop_frames)
        
        def process_batch(batch_indices):
            batch_results = []
            for idx in batch_indices:
                result = self._process_single_frame(
                    crop_frames[idx], full_img.copy(), ox1, oy1, ox2, oy2
                )
                batch_results.append((idx, result))
            return batch_results
        
        # Create batches
        batches = []
        for i in range(0, len(crop_frames), self.batch_size):
            batch = list(range(i, min(i + self.batch_size, len(crop_frames))))
            batches.append(batch)
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_batch = {executor.submit(process_batch, batch): batch for batch in batches}
            
            for future in tqdm(as_completed(future_to_batch), total=len(batches), desc='Fast Paste:'):
                batch_results = future.result()
                for idx, result in batch_results:
                    processed_frames[idx] = result
        
        return processed_frames
    
    def _sequential_process_frames(self, crop_frames, full_img, ox1, oy1, ox2, oy2, frame_w, frame_h):
        """Process frames sequentially"""
        processed_frames = []
        
        for crop_frame in tqdm(crop_frames, 'Fast Paste:'):
            result = self._process_single_frame(
                crop_frame, full_img.copy(), ox1, oy1, ox2, oy2
            )
            processed_frames.append(result)
        
        return processed_frames
    
    def _process_single_frame(self, crop_frame, full_img, ox1, oy1, ox2, oy2):
        """Process a single frame with selected blending method"""
        # Resize crop frame
        resized_frame = cv2.resize(crop_frame.astype(np.uint8), (ox2-ox1, oy2-oy1))
        
        if self.blend_method == "seamless":
            # Use original seamless clone (slowest but best quality)
            mask = 255 * np.ones(resized_frame.shape, resized_frame.dtype)
            location = ((ox1+ox2) // 2, (oy1+oy2) // 2)
            return cv2.seamlessClone(resized_frame, full_img, mask, location, cv2.NORMAL_CLONE)
        
        elif self.blend_method == "simple":
            # Fast simple blending
            mask = 255 * np.ones(resized_frame.shape[:2], dtype=np.uint8)
            location = ((ox1+ox2) // 2, (oy1+oy2) // 2)
            return FastSeamlessClone.simple_blend(resized_frame, full_img, mask, location)
        
        elif self.blend_method == "gaussian":
            # Gaussian blending
            location = ((ox1+ox2) // 2, (oy1+oy2) // 2)
            return FastSeamlessClone.gaussian_blend(resized_frame, full_img, location)
        
        elif self.blend_method == "feather":
            # Feathered blending
            location = ((ox1+ox2) // 2, (oy1+oy2) // 2)
            return FastSeamlessClone.feather_blend(resized_frame, full_img, location)
        
        else:
            # Direct replacement (fastest)
            full_img[oy1:oy2, ox1:ox2] = resized_frame
            return full_img
    
    def _save_processed_video(self, processed_frames, original_video_path, audio_path, output_path):
        """Save processed frames to video"""
        # Get original video properties
        cap = cv2.VideoCapture(original_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        if not processed_frames:
            raise ValueError("No processed frames to save")
        
        frame_h, frame_w = processed_frames[0].shape[:2]
        
        # Create temporary video file
        tmp_path = str(uuid.uuid4()) + '.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(tmp_path, fourcc, fps, (frame_w, frame_h))
        
        for frame in processed_frames:
            out.write(frame)
        out.release()
        
        # Add audio and save final video
        save_video_with_watermark(tmp_path, audio_path, output_path, watermark=False)
        os.remove(tmp_path)


# GPU-accelerated version (if available)
class GPUOptimizedPastePic:
    """GPU-accelerated paste operations using OpenCV's GPU functions"""
    
    def __init__(self):
        self.use_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if self.use_gpu:
            print("GPU acceleration available for paste operations")
        else:
            print("GPU acceleration not available, falling back to CPU")
    
    def paste_video_gpu(self, video_path, pic_path, crop_info, new_audio_path, full_video_path, extended_crop=False):
        """GPU-accelerated paste operation"""
        if not self.use_gpu:
            # Fall back to CPU version
            optimizer = OptimizedPastePic("high", "simple")
            return optimizer.paste_video(video_path, pic_path, crop_info, new_audio_path, full_video_path, extended_crop)
        
        # GPU implementation would go here
        # This requires careful management of GPU memory and CV2 CUDA operations
        pass


# Streaming paste processor for very long videos
class StreamingPasteProcessor:
    """Memory-efficient streaming processor for very long videos"""
    
    def __init__(self, optimization_level="medium", chunk_size=100):
        self.optimization_level = optimization_level
        self.chunk_size = chunk_size
        self.processor = OptimizedPastePic(optimization_level, "simple")
    
    def paste_video_streaming(self, video_path, pic_path, crop_info, new_audio_path, full_video_path, extended_crop=False):
        """Process video in chunks to manage memory usage"""
        
        # Load background
        full_img = self.processor._load_background(pic_path)
        
        # Open video for streaming
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate crop coordinates
        r_w, r_h = crop_info[0]
        clx, cly, crx, cry = crop_info[1] 
        lx, ly, rx, ry = crop_info[2]
        lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
        
        if extended_crop:
            oy1, oy2, ox1, ox2 = cly, cry, clx, crx
        else:
            oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx
        
        # Create temporary output video
        tmp_path = str(uuid.uuid4()) + '.mp4'
        frame_h, frame_w = full_img.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(tmp_path, fourcc, fps, (frame_w, frame_h))
        
        # Process in chunks
        processed_frames = 0
        pbar = tqdm(total=frame_count, desc='Streaming Paste')
        
        while processed_frames < frame_count:
            # Read chunk
            chunk_frames = []
            for _ in range(min(self.chunk_size, frame_count - processed_frames)):
                ret, frame = cap.read()
                if not ret:
                    break
                chunk_frames.append(frame)
            
            if not chunk_frames:
                break
            
            # Process chunk
            for frame in chunk_frames:
                result = self.processor._process_single_frame(
                    frame, full_img.copy(), ox1, oy1, ox2, oy2
                )
                out.write(result)
                processed_frames += 1
                pbar.update(1)
        
        cap.release()
        out.release()
        pbar.close()
        
        # Add audio and save final video
        save_video_with_watermark(tmp_path, new_audio_path, full_video_path, watermark=False)
        os.remove(tmp_path)


# Main optimized paste_pic function
def fast_paste_pic(video_path, pic_path, crop_info, new_audio_path, full_video_path, 
                   extended_crop=False, optimization_level="medium", blend_method="simple"):
    """
    Fast optimized version of paste_pic with multiple modes
    """
    processor = OptimizedPastePic(optimization_level, blend_method)
    return processor.paste_video(video_path, pic_path, crop_info, 
                                 new_audio_path, full_video_path, extended_crop)


def paste_pic(video_path, pic_path, crop_info, new_audio_path, full_video_path, extended_crop=False):
    """
    Standard paste_pic function - wrapper around fast_paste_pic for compatibility
    """
    return fast_paste_pic(video_path, pic_path, crop_info, new_audio_path, 
                         full_video_path, extended_crop=extended_crop, 
                         optimization_level="medium", blend_method="simple")