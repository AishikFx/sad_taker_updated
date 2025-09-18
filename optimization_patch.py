# Optimization patch for SadTalker GPU utilization
# This shows the key changes needed to improve GPU usage

def make_animation_optimized(source_image, source_semantics, target_semantics,
                            generator, kp_detector, he_estimator, mapping, 
                            yaw_c_seq=None, pitch_c_seq=None, roll_c_seq=None,
                            use_exp=True, batch_size=8):
    """
    Optimized version that processes multiple frames in parallel
    """
    with torch.no_grad():
        # Pre-compute source keypoints once
        kp_canonical = kp_detector(source_image)
        he_source = mapping(source_semantics)
        kp_source = keypoint_transformation(kp_canonical, he_source)
        
        total_frames = target_semantics.shape[1]
        predictions = []
        
        # Process frames in batches instead of one by one
        for start_idx in tqdm(range(0, total_frames, batch_size), 'Face Renderer (Batched):'):
            end_idx = min(start_idx + batch_size, total_frames)
            batch_frames = end_idx - start_idx
            
            # Prepare batch data
            batch_target_semantics = target_semantics[:, start_idx:end_idx]
            batch_target_semantics = batch_target_semantics.reshape(-1, batch_target_semantics.shape[-1])
            
            # Batch process driving parameters
            he_driving_batch = mapping(batch_target_semantics)
            
            if yaw_c_seq is not None:
                he_driving_batch['yaw_in'] = yaw_c_seq[:, start_idx:end_idx].reshape(-1)
            if pitch_c_seq is not None:
                he_driving_batch['pitch_in'] = pitch_c_seq[:, start_idx:end_idx].reshape(-1)
            if roll_c_seq is not None:
                he_driving_batch['roll_in'] = roll_c_seq[:, start_idx:end_idx].reshape(-1)
            
            # Batch keypoint transformation
            kp_driving_batch = keypoint_transformation(kp_canonical.repeat(batch_frames, 1, 1), he_driving_batch)
            
            # Batch generation - this is where we gain massive speedup
            source_image_batch = source_image.repeat(batch_frames, 1, 1, 1)
            kp_source_batch = {k: v.repeat(batch_frames, 1, 1) for k, v in kp_source.items()}
            
            out_batch = generator(source_image_batch, kp_source=kp_source_batch, kp_driving=kp_driving_batch)
            
            # Add batch predictions
            batch_predictions = out_batch['prediction'].reshape(1, batch_frames, *out_batch['prediction'].shape[1:])
            predictions.append(batch_predictions)
        
        predictions_ts = torch.cat(predictions, dim=1)
    return predictions_ts

# Recommended settings for different GPU memory sizes:
GPU_MEMORY_CONFIGS = {
    "8GB": {"batch_size": 4, "frame_batch_size": 8},
    "12GB": {"batch_size": 6, "frame_batch_size": 16}, 
    "16GB": {"batch_size": 8, "frame_batch_size": 24},
    "24GB": {"batch_size": 12, "frame_batch_size": 32}
}