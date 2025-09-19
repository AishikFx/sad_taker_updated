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

from src.facerender.modules.keypoint_detector import HEEstimator, KPDetector
from src.facerender.modules.mapping import MappingNet
from src.facerender.modules.generator import OcclusionAwareGenerator, OcclusionAwareSPADEGenerator
from src.facerender.modules.make_animation import make_animation
# Import optimized versions
from src.facerender.modules.make_animation_fast import make_animation_fast
from src.utils.fast_seamless_clone import fast_seamless_clone, process_video_frames_fast 

from pydub import AudioSegment 
# Import optimized face enhancer functions
from src.utils.face_enhancer import fast_enhancer_generator_with_len, fast_enhancer_list
from src.utils.paste_pic import paste_pic
# Import optimized paste functions  
from src.utils.paste_pic import fast_paste_pic, OptimizedPastePic
from src.utils.videoio import save_video_with_watermark

try:
    import webui  # in webui
    in_webui = True
except:
    in_webui = False

class AnimateFromCoeff():

    def __init__(self, sadtalker_path, device, optimization_level="medium"):
        # Keep the device optimization from new code but don't over-optimize
        self.use_cuda = torch.cuda.is_available() and device != 'cpu'
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.optimization_level = optimization_level
        
        print(f"Initializing AnimateFromCoeff on device: {self.device}")
        print(f"Optimization level: {optimization_level}")

        with open(sadtalker_path['facerender_yaml']) as f:
            config = yaml.safe_load(f)

        generator = OcclusionAwareSPADEGenerator(**config['model_params']['generator_params'],
                                                    **config['model_params']['common_params'])
        kp_extractor = KPDetector(**config['model_params']['kp_detector_params'],
                                    **config['model_params']['common_params'])
        he_estimator = HEEstimator(**config['model_params']['he_estimator_params'],
                               **config['model_params']['common_params'])
        mapping = MappingNet(**config['model_params']['mapping_params'])

        generator.to(self.device)
        kp_extractor.to(self.device)
        he_estimator.to(self.device)
        mapping.to(self.device)
        
        for param in generator.parameters():
            param.requires_grad = False
        for param in kp_extractor.parameters():
            param.requires_grad = False 
        for param in he_estimator.parameters():
            param.requires_grad = False
        for param in mapping.parameters():
            param.requires_grad = False

        # Enable some optimizations but not aggressively
        if self.use_cuda:
            torch.backends.cudnn.benchmark = True

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

        checkpoint = safetensors.torch.load_file(checkpoint_path)

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
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
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
        checkpoint = torch.load(checkpoint_path,  map_location=torch.device(device))
        if mapping is not None:
            mapping.load_state_dict(checkpoint['mapping'])
        if discriminator is not None:
            discriminator.load_state_dict(checkpoint['discriminator'])
        if optimizer_mapping is not None:
            optimizer_mapping.load_state_dict(checkpoint['optimizer_mapping'])
        if optimizer_discriminator is not None:
            optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])

        return checkpoint['epoch']

    def generate(self, x, video_save_dir, pic_path, crop_info, enhancer=None, background_enhancer=None, preprocess='crop', img_size=256, profile=False, batch_size=1, optimization_level=None, **kwargs):
        
        # Use instance optimization level if not overridden
        if optimization_level is None:
            optimization_level = self.optimization_level

        # Use the old, working tensor processing approach
        source_image=x['source_image'].type(torch.FloatTensor)
        source_semantics=x['source_semantics'].type(torch.FloatTensor)
        target_semantics=x['target_semantics_list'].type(torch.FloatTensor) 
        source_image=source_image.to(self.device)
        source_semantics=source_semantics.to(self.device)
        target_semantics=target_semantics.to(self.device)
        
        if 'yaw_c_seq' in x:
            yaw_c_seq = x['yaw_c_seq'].type(torch.FloatTensor)
            yaw_c_seq = yaw_c_seq.to(self.device)
        else:
            yaw_c_seq = None
        if 'pitch_c_seq' in x:
            pitch_c_seq = x['pitch_c_seq'].type(torch.FloatTensor)
            pitch_c_seq = pitch_c_seq.to(self.device)
        else:
            pitch_c_seq = None
        if 'roll_c_seq' in x:
            roll_c_seq = x['roll_c_seq'].type(torch.FloatTensor)
            roll_c_seq = roll_c_seq.to(self.device)
        else:
            roll_c_seq = None

        frame_num = x['frame_num']

        # Use optimized make_animation based on optimization level
        if optimization_level in ["high", "extreme"]:
            print(f"Using fast Face Renderer (optimization: {optimization_level})")
            predictions_video = make_animation_fast(source_image, source_semantics, target_semantics,
                                            self.generator, self.kp_extractor, self.he_estimator, self.mapping, 
                                            yaw_c_seq, pitch_c_seq, roll_c_seq, use_exp = True, 
                                            optimization_level=optimization_level)
        else:
            # Keep the original make_animation call for compatibility
            predictions_video = make_animation(source_image, source_semantics, target_semantics,
                                            self.generator, self.kp_extractor, self.he_estimator, self.mapping, 
                                            yaw_c_seq, pitch_c_seq, roll_c_seq, use_exp = True)

        # Use the original, working tensor-to-numpy conversion
        predictions_video = predictions_video.reshape((-1,)+predictions_video.shape[2:])
        predictions_video = predictions_video[:frame_num]

        video = []
        for idx in range(predictions_video.shape[0]):
            image = predictions_video[idx]
            image = np.transpose(image.data.cpu().numpy(), [1, 2, 0]).astype(np.float32)
            video.append(image)
        result = img_as_ubyte(video)

        ### the generated video is 256x256, so we keep the aspect ratio, 
        original_size = crop_info[0]
        if original_size:
            result = [ cv2.resize(result_i,(img_size, int(img_size * original_size[1]/original_size[0]) )) for result_i in result ]
        
        video_name = x['video_name']  + '.mp4'
        path = os.path.join(video_save_dir, 'temp_'+video_name)
        
        imageio.mimsave(path, result,  fps=float(25))

        av_path = os.path.join(video_save_dir, video_name)
        return_path = av_path 
        
        audio_path =  x['audio_path'] 
        audio_name = os.path.splitext(os.path.split(audio_path)[-1])[0]
        new_audio_path = os.path.join(video_save_dir, audio_name+'.wav')
        start_time = 0
        # cog will not keep the .mp3 filename
        sound = AudioSegment.from_file(audio_path)
        frames = frame_num 
        end_time = start_time + frames*1/25*1000
        word1=sound.set_frame_rate(16000)
        word = word1[start_time:end_time]
        word.export(new_audio_path, format="wav")

        save_video_with_watermark(path, new_audio_path, av_path, watermark= False)
        print(f'The generated video is named {video_save_dir}/{video_name}') 

        # CRITICAL: This is where the full video gets created - use optimized paste if available
        if 'full' in preprocess.lower():
            # only add watermark to the full image.
            video_name_full = x['video_name']  + '_full.mp4'
            full_video_path = os.path.join(video_save_dir, video_name_full)
            return_path = full_video_path
            
            # Use optimized paste based on optimization level
            if optimization_level in ["high", "extreme"]:
                print(f"Using fast paste processing (optimization: {optimization_level})")
                try:
                    # Try optimized paste first
                    optimizer = OptimizedPastePic(optimization_level=optimization_level, 
                                                blend_method="fast" if optimization_level == "extreme" else "balanced")
                    optimizer.paste_video(path, pic_path, crop_info, new_audio_path, full_video_path, 
                                        extended_crop=True if 'ext' in preprocess.lower() else False)
                except Exception as e:
                    print(f"Fast paste failed ({e}), falling back to original paste_pic")
                    paste_pic(path, pic_path, crop_info, new_audio_path, full_video_path, 
                            extended_crop=True if 'ext' in preprocess.lower() else False)
            else:
                paste_pic(path, pic_path, crop_info, new_audio_path, full_video_path, 
                        extended_crop=True if 'ext' in preprocess.lower() else False)
            print(f'The generated video is named {video_save_dir}/{video_name_full}') 
        else:
            full_video_path = av_path 

        #### paste back then enhancers
        if enhancer:
            video_name_enhancer = x['video_name']  + '_enhanced.mp4'
            enhanced_path = os.path.join(video_save_dir, 'temp_'+video_name_enhancer)
            av_path_enhancer = os.path.join(video_save_dir, video_name_enhancer) 
            return_path = av_path_enhancer

            # Use optimized face enhancer based on optimization level  
            if optimization_level in ["high", "extreme"]:
                print(f"Using fast face enhancer (optimization: {optimization_level})")
                try:
                    enhanced_images_gen_with_len = fast_enhancer_generator_with_len(
                        full_video_path, method=enhancer, bg_upsampler=background_enhancer,
                        optimization_level=optimization_level, batch_size=8 if optimization_level == "extreme" else 4)
                    imageio.mimsave(enhanced_path, enhanced_images_gen_with_len, fps=float(25))
                except Exception as e:
                    print(f"Fast enhancer failed ({e}), falling back to original enhancer")
                    try:
                        enhanced_images_gen_with_len = fast_enhancer_generator_with_len(full_video_path, method=enhancer, bg_upsampler=background_enhancer)
                        imageio.mimsave(enhanced_path, enhanced_images_gen_with_len, fps=float(25))
                    except:
                        enhanced_images_gen_with_len = fast_enhancer_list(full_video_path, method=enhancer, bg_upsampler=background_enhancer)
                        imageio.mimsave(enhanced_path, enhanced_images_gen_with_len, fps=float(25))
            else:
                try:
                    enhanced_images_gen_with_len = fast_enhancer_generator_with_len(full_video_path, method=enhancer, bg_upsampler=background_enhancer)
                    imageio.mimsave(enhanced_path, enhanced_images_gen_with_len, fps=float(25))
                except:
                    enhanced_images_gen_with_len = fast_enhancer_list(full_video_path, method=enhancer, bg_upsampler=background_enhancer)
                    imageio.mimsave(enhanced_path, enhanced_images_gen_with_len, fps=float(25))
            
            save_video_with_watermark(enhanced_path, new_audio_path, av_path_enhancer, watermark= False)
            print(f'The generated video is named {video_save_dir}/{video_name_enhancer}')
            os.remove(enhanced_path)

        os.remove(path)
        os.remove(new_audio_path)

        # Only do gentle GPU cleanup at the very end, after all processing is complete
        if self.use_cuda:
            torch.cuda.empty_cache()

        return return_path