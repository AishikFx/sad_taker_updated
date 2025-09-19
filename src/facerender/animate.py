import os
import cv2
import yaml
import numpy as np
import warnings
from skimage import img_as_ubyte
import safetensors
import safetensors.torch 
warnings.filterwarnings('ignore')

import imageio
import torch
import torchvision
import time

from src.facerender.modules.keypoint_detector import HEEstimator, KPDetector
from src.facerender.modules.mapping import MappingNet
from src.facerender.modules.generator import  OcclusionAwareSPADEGenerator
from src.facerender.modules.make_animation import make_animation 

from pydub import AudioSegment 
from src.utils.face_enhancer import enhancer_generator_with_len, enhancer_list
from src.utils.paste_pic import paste_pic
from src.utils.videoio import save_video_with_watermark

try:
    in_webui = True
except:
    in_webui = False

class AnimateFromCoeff():

    def __init__(self, sadtalker_path, device):
        # Determine optimal device usage
        self.use_cuda = torch.cuda.is_available() and device != 'cpu'
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        
        print(f"Initializing AnimateFromCoeff on device: {self.device}")

        with open(sadtalker_path['facerender_yaml']) as f:
            config = yaml.safe_load(f)

        generator = OcclusionAwareSPADEGenerator(**config['model_params']['generator_params'],
                                                    **config['model_params']['common_params'])
        kp_extractor = KPDetector(**config['model_params']['kp_detector_params'],
                                    **config['model_params']['common_params'])
        he_estimator = HEEstimator(**config['model_params']['he_estimator_params'],
                               **config['model_params']['common_params'])
        mapping = MappingNet(**config['model_params']['mapping_params'])

        # Move models to device and set memory optimization for CUDA
        generator.to(self.device)
        kp_extractor.to(self.device)
        he_estimator.to(self.device)
        mapping.to(self.device)
        
        # Disable gradients for inference
        for param in generator.parameters():
            param.requires_grad = False
        for param in kp_extractor.parameters():
            param.requires_grad = False 
        for param in he_estimator.parameters():
            param.requires_grad = False
        for param in mapping.parameters():
            param.requires_grad = False

        # Enable memory optimization for CUDA
        if self.use_cuda:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        if sadtalker_path is not None:
            if 'checkpoint' in sadtalker_path: # use safe tensor
                self.load_cpk_facevid2vid_safetensor(sadtalker_path['checkpoint'], kp_detector=kp_extractor, generator=generator, he_estimator=None)
            else:
                self.load_cpk_facevid2vid(sadtalker_path['free_view_checkpoint'], kp_detector=kp_extractor, generator=generator, he_estimator=he_estimator)
        else:
            raise AttributeError("Checkpoint should be specified for video head pose estimator.")

        if  sadtalker_path['mappingnet_checkpoint'] is not None:
            self.load_cpk_mapping(sadtalker_path['mappingnet_checkpoint'], mapping=mapping)
        else:
            raise AttributeError("Checkpoint should be specified for video head pose estimator.") 

        self.kp_extractor = kp_extractor
        self.generator = generator
        self.he_estimator = he_estimator
        self.mapping = mapping

        self.kp_extractor.eval()
        self.generator.eval()
        self.he_estimator.eval()
        self.mapping.eval()
    
    def load_cpk_facevid2vid_safetensor(self, checkpoint_path, generator=None, 
                        kp_detector=None, he_estimator=None,  
                        device="cpu"):

        # Load directly to the target device to avoid CPU->GPU transfer
        map_location = self.device if hasattr(self, 'device') else device
        checkpoint = safetensors.torch.load_file(checkpoint_path, device=str(map_location))

        if generator is not None:
            x_generator = {}
            for k,v in checkpoint.items():
                if 'generator' in k:
                    x_generator[k.replace('generator.', '')] = v
            generator.load_state_dict(x_generator)
        if kp_detector is not None:
            x_generator = {}
            for k,v in checkpoint.items():
                if 'kp_extractor' in k:
                    x_generator[k.replace('kp_extractor.', '')] = v
            kp_detector.load_state_dict(x_generator)
        if he_estimator is not None:
            x_generator = {}
            for k,v in checkpoint.items():
                if 'he_estimator' in k:
                    x_generator[k.replace('he_estimator.', '')] = v
            he_estimator.load_state_dict(x_generator)
        
        return None

    def load_cpk_facevid2vid(self, checkpoint_path, generator=None, discriminator=None, 
                        kp_detector=None, he_estimator=None, optimizer_generator=None, 
                        optimizer_discriminator=None, optimizer_kp_detector=None, 
                        optimizer_he_estimator=None, device="cpu"):
        
        # Load directly to the target device
        map_location = self.device if hasattr(self, 'device') else device
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        if generator is not None:
            generator.load_state_dict(checkpoint['generator'])
        if kp_detector is not None:
            kp_detector.load_state_dict(checkpoint['kp_detector'])
        if he_estimator is not None:
            he_estimator.load_state_dict(checkpoint['he_estimator'])
        if discriminator is not None:
            try:
               discriminator.load_state_dict(checkpoint['discriminator'])
            except:
               print ('No discriminator in the state-dict. Dicriminator will be randomly initialized')
        if optimizer_generator is not None:
            optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
        if optimizer_discriminator is not None:
            try:
                optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
            except RuntimeError as e:
                print ('No discriminator optimizer in the state-dict. Optimizer will be not initialized')
        if optimizer_kp_detector is not None:
            optimizer_kp_detector.load_state_dict(checkpoint['optimizer_kp_detector'])
        if optimizer_he_estimator is not None:
            optimizer_he_estimator.load_state_dict(checkpoint['optimizer_he_estimator'])

        return checkpoint['epoch']
    
    def load_cpk_mapping(self, checkpoint_path, mapping=None, discriminator=None,
                 optimizer_mapping=None, optimizer_discriminator=None, device='cpu'):
        
        # Load directly to the target device
        map_location = self.device if hasattr(self, 'device') else device
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        if mapping is not None:
            mapping.load_state_dict(checkpoint['mapping'])
        if discriminator is not None:
            discriminator.load_state_dict(checkpoint['discriminator'])
        if optimizer_mapping is not None:
            optimizer_mapping.load_state_dict(checkpoint['optimizer_mapping'])
        if optimizer_discriminator is not None:
            optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])

        return checkpoint['epoch']

    def _optimize_tensor_operations(self, tensor_data):
        """Optimize tensor operations based on device capabilities"""
        if self.use_cuda and tensor_data.is_cuda:
            # Use faster CUDA operations
            with torch.cuda.amp.autocast(enabled=True):
                return tensor_data
        return tensor_data

    def _efficient_tensor_to_numpy(self, predictions_video, frame_num, target_size=None):
        """Efficiently convert tensor to numpy with GPU resizing if needed"""
        predictions_video = predictions_video.reshape((-1,)+predictions_video.shape[2:])
        predictions_video = predictions_video[:frame_num]

        # GPU-based resizing if target size is specified
        if target_size is not None and self.use_cuda:
            h, w = target_size
            # Reshape to [T, C, H, W] for interpolation
            predictions_video = torch.nn.functional.interpolate(
                predictions_video, size=(h, w), mode='bilinear', align_corners=False
            )

        if self.use_cuda:
            # Minimize GPU-CPU transfers by doing operations on GPU first
            with torch.inference_mode():
                # Keep tensor operations on GPU as long as possible
                predictions_video = predictions_video.detach()
                # Single transfer to CPU
                video_np = predictions_video.cpu().numpy()
        else:
            # CPU operations
            with torch.inference_mode():
                video_np = predictions_video.detach().numpy()

        video = []
        for idx in range(video_np.shape[0]):
            image = np.transpose(video_np[idx], [1, 2, 0]).astype(np.float32)
            video.append(image)
        
        return img_as_ubyte(video)

    def generate(self, x, video_save_dir, pic_path, crop_info, enhancer=None, background_enhancer=None, preprocess='crop', img_size=256, profile=False, batch_size=8, enhancer_batch_threshold=None):

        # Optimize tensor loading and device placement
        source_image = x['source_image'].type(torch.FloatTensor).to(self.device, non_blocking=self.use_cuda)
        source_semantics = x['source_semantics'].type(torch.FloatTensor).to(self.device, non_blocking=self.use_cuda)
        target_semantics = x['target_semantics_list'].type(torch.FloatTensor).to(self.device, non_blocking=self.use_cuda)
        
        # Handle optional sequences with optimized device placement
        yaw_c_seq = None
        pitch_c_seq = None
        roll_c_seq = None
        
        if 'yaw_c_seq' in x:
            yaw_c_seq = x['yaw_c_seq'].type(torch.FloatTensor).to(self.device, non_blocking=self.use_cuda)
        if 'pitch_c_seq' in x:
            pitch_c_seq = x['pitch_c_seq'].type(torch.FloatTensor).to(self.device, non_blocking=self.use_cuda)
        if 'roll_c_seq' in x:
            roll_c_seq = x['roll_c_seq'].type(torch.FloatTensor).to(self.device, non_blocking=self.use_cuda)

        frame_num = x['frame_num']

        # Synchronization and profiling
        if profile and self.use_cuda:
            torch.cuda.synchronize()
        t_make0 = time.time()
        
        # Core GPU-heavy step with context management
        if self.use_cuda:
            with torch.cuda.amp.autocast(enabled=False):  # Disable if causing issues
                with torch.no_grad():
                    predictions_video = make_animation(source_image, source_semantics, target_semantics,
                                                    self.generator, self.kp_extractor, self.he_estimator, self.mapping, 
                                                    yaw_c_seq, pitch_c_seq, roll_c_seq, use_exp = True)
        else:
            with torch.no_grad():
                predictions_video = make_animation(source_image, source_semantics, target_semantics,
                                                self.generator, self.kp_extractor, self.he_estimator, self.mapping, 
                                                yaw_c_seq, pitch_c_seq, roll_c_seq, use_exp = True)
        
        if profile and self.use_cuda:
            torch.cuda.synchronize()
        t_make1 = time.time()
        if profile:
            device_info = f"({'CUDA' if self.use_cuda else 'CPU'})"
            print(f'[profile] make_animation {device_info} time: {t_make1 - t_make0:.3f}s')

        # Efficient tensor to numpy conversion
        result = self._efficient_tensor_to_numpy(predictions_video, frame_num)

        # Clear GPU memory if using CUDA
        if self.use_cuda:
            del predictions_video, source_image, source_semantics, target_semantics
            if yaw_c_seq is not None:
                del yaw_c_seq
            if pitch_c_seq is not None:
                del pitch_c_seq
            if roll_c_seq is not None:
                del roll_c_seq
            torch.cuda.empty_cache()

        ### the generated video is 256x256, so we keep the aspect ratio, 
        original_size = crop_info[0]
        if original_size:
            result = [ cv2.resize(result_i,(img_size, int(img_size * original_size[1]/original_size[0]) )) for result_i in result ]
        
        video_name = x['video_name']  + '.mp4'
        path = os.path.join(video_save_dir, 'temp_'+video_name)
        
        t_save0 = time.time()
        imageio.mimsave(path, result, fps=float(25))
        t_save1 = time.time()
        if profile:
            print(f'[profile] save raw video frames to mp4 time: {t_save1 - t_save0:.3f}s')

        av_path = os.path.join(video_save_dir, video_name)
        return_path = av_path 
        
        audio_path = x['audio_path'] 
        audio_name = os.path.splitext(os.path.split(audio_path)[-1])[0]
        new_audio_path = os.path.join(video_save_dir, audio_name+'.wav')
        start_time = 0
        
        # Audio processing
        t_audio0 = time.time()
        sound = AudioSegment.from_file(audio_path)
        frames = frame_num 
        end_time = start_time + frames*1/25*1000
        word1 = sound.set_frame_rate(16000)
        word = word1[start_time:end_time]
        word.export(new_audio_path, format="wav")
        t_audio1 = time.time()
        if profile:
            print(f'[profile] audio slicing/export time: {t_audio1 - t_audio0:.3f}s')

        t_mux0 = time.time()
        save_video_with_watermark(path, new_audio_path, av_path, watermark=False)
        t_mux1 = time.time()
        if profile:
            print(f'[profile] mux audio+video (watermark) time: {t_mux1 - t_mux0:.3f}s')
        print(f'The generated video is named {video_save_dir}/{video_name}') 

        if 'full' in preprocess.lower():
            # only add watermark to the full image.
            video_name_full = x['video_name']  + '_full.mp4'
            full_video_path = os.path.join(video_save_dir, video_name_full)
            return_path = full_video_path
            paste_pic(path, pic_path, crop_info, new_audio_path, full_video_path, extended_crop= True if 'ext' in preprocess.lower() else False)
            print(f'The generated video is named {video_save_dir}/{video_name_full}') 
        else:
            full_video_path = av_path 

        #### paste back then enhancers
        if enhancer:
            video_name_enhancer = x['video_name']  + '_enhanced.mp4'
            enhanced_path = os.path.join(video_save_dir, 'temp_'+video_name_enhancer)
            av_path_enhancer = os.path.join(video_save_dir, video_name_enhancer) 
            return_path = av_path_enhancer
            
            # choose enhancer mode automatically for short clips
            try:
                if enhancer_batch_threshold is None:
                    threshold = int(os.environ.get('ENHANCER_BATCH_THRESHOLD', '128'))
                else:
                    threshold = int(enhancer_batch_threshold)
            except Exception:
                threshold = 128

            # Optimize batch size based on device and memory
            if self.use_cuda:
                # For GPU, use larger batch sizes for efficiency
                chosen_batch_size = batch_size if frame_num >= threshold else min(batch_size, 4)
            else:
                # For CPU, use smaller batch sizes to avoid memory issues
                chosen_batch_size = min(batch_size, 2) if frame_num >= threshold else 1

            try:
                t_enh0 = time.time()
                enhanced_images_gen_with_len = enhancer_generator_with_len(full_video_path, method=enhancer, bg_upsampler=background_enhancer, batch_size=chosen_batch_size)
                imageio.mimsave(enhanced_path, enhanced_images_gen_with_len, fps=float(25))
                t_enh1 = time.time()
                if profile:
                    device_info = f"({'CUDA' if self.use_cuda else 'CPU'})"
                    print(f'[profile] enhancer_generator_with_len + save {device_info} time: {t_enh1 - t_enh0:.3f}s (requested_batch={batch_size}, chosen_batch={chosen_batch_size}, threshold={threshold})')
            except Exception:
                t_enh0 = time.time()
                enhanced_images_gen_with_len = enhancer_list(full_video_path, method=enhancer, bg_upsampler=background_enhancer, batch_size=chosen_batch_size)
                imageio.mimsave(enhanced_path, enhanced_images_gen_with_len, fps=float(25))
                t_enh1 = time.time()
                if profile:
                    device_info = f"({'CUDA' if self.use_cuda else 'CPU'})"
                    print(f'[profile] enhancer_list + save {device_info} time: {t_enh1 - t_enh0:.3f}s (requested_batch={batch_size}, chosen_batch={chosen_batch_size}, threshold={threshold})')
            
            save_video_with_watermark(enhanced_path, new_audio_path, av_path_enhancer, watermark=False)
            print(f'The generated video is named {video_save_dir}/{video_name_enhancer}')
            os.remove(enhanced_path)

        os.remove(path)
        os.remove(new_audio_path)

        # Final GPU memory cleanup
        if self.use_cuda:
            torch.cuda.empty_cache()

        return return_path