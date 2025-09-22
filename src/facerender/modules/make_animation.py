import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm 

def compute_keypoint_area_torch(keypoints_value):
    """
    Compute approximate area of keypoints using PyTorch only (no SciPy).
    Uses bounding box area as a fast approximation to ConvexHull volume.
    
    Args:
        keypoints_value: tensor of shape [batch, num_points, 2] or [batch, num_points, 3]
    Returns:
        area: tensor of shape [batch]
    """
    # Handle both 2D and 3D keypoints - use first 2 dimensions for area
    if keypoints_value.shape[-1] > 2:
        keypoints_2d = keypoints_value[..., :2]  # Take x, y coordinates only
    else:
        keypoints_2d = keypoints_value
    
    # Get min/max coordinates for bounding box
    min_coords = torch.min(keypoints_2d, dim=1)[0]  # [batch, 2]
    max_coords = torch.max(keypoints_2d, dim=1)[0]  # [batch, 2]
    
    # Compute bounding box area
    bbox_area = (max_coords - min_coords).prod(dim=1)  # [batch]
    
    # Add small epsilon to avoid zero areas
    bbox_area = torch.clamp(bbox_area, min=1e-8)
    
    return bbox_area


def normalize_kp_production(kp_source, kp_driving, kp_driving_initial, 
                          adapt_movement_scale=False,
                          use_relative_movement=False, 
                          use_relative_jacobian=False,
                          precomputed_scale=None):
    """
    Production-optimized normalize_kp that:
    1. Never leaves GPU (no SciPy)
    2. Uses stable linear algebra 
    3. Minimizes allocations
    4. Supports precomputed scale factor
    """
    
    # Handle movement scale computation
    if adapt_movement_scale and precomputed_scale is None:
        # Use fast PyTorch-only area computation
        source_area = compute_keypoint_area_torch(kp_source['value'])
        driving_area = compute_keypoint_area_torch(kp_driving_initial['value'])
        
        # Avoid division by zero and use stable computation
        eps = 1e-8
        adapt_movement_scale = torch.sqrt(source_area / (driving_area + eps))
    elif precomputed_scale is not None:
        adapt_movement_scale = precomputed_scale
    else:
        adapt_movement_scale = 1.0
    
    # Start with driving keypoints (avoid full dict copy when possible)
    if use_relative_movement or use_relative_jacobian:
        # Only copy dict when we actually modify it
        kp_new = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                  for k, v in kp_driving.items()}
    else:
        # No modification needed, return as-is
        return kp_driving

    if use_relative_movement:
        # Compute relative movement and scale
        kp_value_diff = kp_driving['value'] - kp_driving_initial['value']
        if isinstance(adapt_movement_scale, torch.Tensor):
            # Handle batch dimension properly
            kp_value_diff = kp_value_diff * adapt_movement_scale.unsqueeze(-1).unsqueeze(-1)
        else:
            kp_value_diff = kp_value_diff * adapt_movement_scale
            
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian and 'jacobian' in kp_driving_initial and 'jacobian' in kp_driving and 'jacobian' in kp_source:
            # Only process jacobians if they exist in all keypoint dictionaries
            batch_size, num_points, _, _ = kp_driving_initial['jacobian'].shape
            device = kp_driving_initial['jacobian'].device
            
            try:
                # Use stable solve instead of inverse
                driving_initial_jac = kp_driving_initial['jacobian']
                driving_jac = kp_driving['jacobian']
                
                # Solve: driving_initial_jac @ jacobian_diff = driving_jac
                jacobian_diff = torch.linalg.solve(driving_initial_jac, driving_jac)
                
                # Apply to source jacobian
                kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])
                
            except torch.linalg.LinAlgError:
                # Fallback to pseudoinverse if matrix is singular
                print("⚠️ Warning: Singular matrix in jacobian computation, using pseudoinverse")
                jacobian_diff = torch.matmul(kp_driving['jacobian'], 
                                           torch.linalg.pinv(kp_driving_initial['jacobian']))
                kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])
        elif use_relative_jacobian:
            # Skip jacobian processing if not available
            print("⚠️ Warning: Jacobian processing requested but jacobians not available in keypoints")

    return kp_new


class KeypointNormalizer:
    """
    Stateful normalizer that precomputes scale factors for better performance.
    Use this for batch processing or when source/initial frames don't change.
    """
    
    def __init__(self, kp_source=None, kp_driving_initial=None, 
                 adapt_movement_scale=False):
        self.kp_source = kp_source
        self.kp_driving_initial = kp_driving_initial
        self.precomputed_scale = None
        
        # Debug keypoint shapes and jacobian availability
        if kp_source is not None:
            print(f" KeypointNormalizer: source keypoints shape: {kp_source['value'].shape}")
            if 'jacobian' in kp_source:
                print(f" KeypointNormalizer: source jacobian shape: {kp_source['jacobian'].shape}")
            else:
                print(" KeypointNormalizer: source jacobian not available")
        if kp_driving_initial is not None:
            print(f" KeypointNormalizer: driving_initial keypoints shape: {kp_driving_initial['value'].shape}")
            if 'jacobian' in kp_driving_initial:
                print(f" KeypointNormalizer: driving_initial jacobian shape: {kp_driving_initial['jacobian'].shape}")
            else:
                print(" KeypointNormalizer: driving_initial jacobian not available")
        
        # Precompute scale factor if possible
        if adapt_movement_scale and kp_source is not None and kp_driving_initial is not None:
            try:
                source_area = compute_keypoint_area_torch(kp_source['value'])
                driving_area = compute_keypoint_area_torch(kp_driving_initial['value'])
                eps = 1e-8
                self.precomputed_scale = torch.sqrt(source_area / (driving_area + eps))
                
                # Handle batch dimension for display - take first element or mean
                if self.precomputed_scale.numel() > 1:
                    display_scale = self.precomputed_scale.mean().item()
                    print(f" Precomputed keypoint scale factor (batch avg): {display_scale:.4f}")
                else:
                    print(f" Precomputed keypoint scale factor: {self.precomputed_scale.item():.4f}")
            except Exception as e:
                print(f"⚠️ Warning: Failed to precompute scale factor: {e}")
                print(f"   Falling back to dynamic scale computation")
                self.precomputed_scale = None
    
    def normalize(self, kp_driving, use_relative_movement=False, use_relative_jacobian=False):
        """Fast normalize using precomputed values."""
        return normalize_kp_production(
            self.kp_source, kp_driving, self.kp_driving_initial,
            adapt_movement_scale=(self.precomputed_scale is not None),
            use_relative_movement=use_relative_movement,
            use_relative_jacobian=use_relative_jacobian,
            precomputed_scale=self.precomputed_scale
        )

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
    
    # Check if jacobian is available - it might not be if estimate_jacobian=False in KPDetector
    if 'jacobian' in kp_canonical:
        jacobian = kp_canonical['jacobian'] # (bs, k, 3, 3)
    else:
        # Create identity jacobian if not available
        batch_size, num_kp = kp.shape[:2]
        device = kp.device
        jacobian = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_kp, 1, 1)

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
    jacobian_transformed = torch.matmul(rot_mat.unsqueeze(1), jacobian)

    # Return a dictionary containing both the transformed values and jacobians.
    return {'value': kp_transformed, 'jacobian': jacobian_transformed}



def make_animation(source_image, source_semantics, target_semantics,
                            generator, kp_detector, he_estimator, mapping, 
                            yaw_c_seq=None, pitch_c_seq=None, roll_c_seq=None,
                            use_exp=True):
    with torch.no_grad():
        predictions = []

        kp_canonical = kp_detector(source_image)
        he_source = mapping(source_semantics)
        kp_source = keypoint_transformation(kp_canonical, he_source)
    
        for frame_idx in tqdm(range(target_semantics.shape[1]), 'Face Renderer:'):
            # still check the dimension
            # print(target_semantics.shape, source_semantics.shape)
            target_semantics_frame = target_semantics[:, frame_idx]
            he_driving = mapping(target_semantics_frame)
            if yaw_c_seq is not None:
                he_driving['yaw_in'] = yaw_c_seq[:, frame_idx]
            if pitch_c_seq is not None:
                he_driving['pitch_in'] = pitch_c_seq[:, frame_idx] 
            if roll_c_seq is not None:
                he_driving['roll_in'] = roll_c_seq[:, frame_idx] 
            
            kp_driving = keypoint_transformation(kp_canonical, he_driving)
                
            kp_norm = kp_driving
            out = generator(source_image, kp_source=kp_source, kp_driving=kp_norm)
            '''
            source_image_new = out['prediction'].squeeze(1)
            kp_canonical_new =  kp_detector(source_image_new)
            he_source_new = he_estimator(source_image_new) 
            kp_source_new = keypoint_transformation(kp_canonical_new, he_source_new, wo_exp=True)
            kp_driving_new = keypoint_transformation(kp_canonical_new, he_driving, wo_exp=True)
            out = generator(source_image_new, kp_source=kp_source_new, kp_driving=kp_driving_new)
            '''
            predictions.append(out['prediction'])
        predictions_ts = torch.stack(predictions, dim=1)
    return predictions_ts

class AnimateModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, generator, kp_extractor, mapping):
        super(AnimateModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.mapping = mapping

        self.kp_extractor.eval()
        self.generator.eval()
        self.mapping.eval()

    def forward(self, x):
        
        source_image = x['source_image']
        source_semantics = x['source_semantics']
        target_semantics = x['target_semantics']
        yaw_c_seq = x['yaw_c_seq']
        pitch_c_seq = x['pitch_c_seq']
        roll_c_seq = x['roll_c_seq']

        predictions_video = make_animation(source_image, source_semantics, target_semantics,
                                        self.generator, self.kp_extractor,
                                        self.mapping, use_exp = True,
                                        yaw_c_seq=yaw_c_seq, pitch_c_seq=pitch_c_seq, roll_c_seq=roll_c_seq)
        
        return predictions_video