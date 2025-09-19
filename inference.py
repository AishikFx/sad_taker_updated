from glob import glob
import shutil
import torch
from time import  strftime
import os, sys, time
from argparse import ArgumentParser

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
from src.utils.optimization_config import OptimizationConfig, PerformanceMonitor

def main(args):
    # Initialize optimization configuration
    if hasattr(args, 'optimization_preset') and args.optimization_preset:
        opt_config = OptimizationConfig(args.optimization_preset)
        opt_config.print_performance_estimate()
        optimization_level = opt_config.get_optimization_level()
    else:
        optimization_level = getattr(args, 'optimization_level', 'medium')
    
    # Initialize performance monitor if profiling
    monitor = PerformanceMonitor() if getattr(args, 'profile', False) else None
    
    # Enable optimizations for better GPU utilization
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Enable mixed precision for faster inference on supported GPUs
    use_amp = torch.cuda.is_available() and args.device == "cuda"
    
    # Disable gradients for inference to save memory
    torch.set_grad_enabled(False)

    pic_path = args.source_image
    audio_path = args.driven_audio
    save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)
    pose_style = args.pose_style
    device = args.device
    batch_size = args.batch_size
    input_yaw_list = args.input_yaw
    input_pitch_list = args.input_pitch
    input_roll_list = args.input_roll
    ref_eyeblink = args.ref_eyeblink
    ref_pose = args.ref_pose

    # More reliable path calculation
    current_root_path = os.path.dirname(os.path.abspath(__file__))

    sadtalker_paths = init_path(args.checkpoint_dir, os.path.join(current_root_path, 'src/config'), args.size, args.old_version, args.preprocess)

    #init model
    preprocess_model = CropAndExtract(sadtalker_paths, device)

    audio_to_coeff = Audio2Coeff(sadtalker_paths,  device)
    
    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device, optimization_level=optimization_level)

    # profiling
    profile = getattr(args, 'profile', False)

    #crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    
    # GPU synchronization for accurate profiling
    if profile and torch.cuda.is_available():
        torch.cuda.synchronize()
    if monitor:
        monitor.start_timing("3DMM Extraction (source image)")
    
    # Resolve and log the pic_path being used for preprocessing
    resolved_pic_path = os.path.abspath(os.path.expanduser(pic_path))
    print(f'Using source image path (resolved): {resolved_pic_path}')
    first_coeff_path, crop_pic_path, crop_info =  preprocess_model.generate(resolved_pic_path, first_frame_dir, args.preprocess,\
                                                                             source_image_flag=True, pic_size=args.size)
    
    if profile and torch.cuda.is_available():
        torch.cuda.synchronize()
    if monitor:
        monitor.end_timing("3DMM Extraction (source image)")
    elif profile:
        t1 = time.time()
        print(f'[profile] 3DMM extraction time: {t1-t0:.3f}s')
    
    if first_coeff_path is None:
        raise RuntimeError("Failed to extract 3DMM coefficients from the input image. Please check the image quality and format.")

    if ref_eyeblink is not None:
        ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print('3DMM Extraction for the reference video providing eye blinking')
        ref_eyeblink_coeff_path, _, _ =  preprocess_model.generate(ref_eyeblink, ref_eyeblink_frame_dir, args.preprocess, source_image_flag=False)
        if ref_eyeblink_coeff_path is None:
            print("Warning: Failed to extract coefficients from reference eye blink video. Proceeding without eye blink reference.")
    else:
        ref_eyeblink_coeff_path=None

    if ref_pose is not None:
        if ref_pose == ref_eyeblink: 
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
            ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            ref_pose_coeff_path, _, _ =  preprocess_model.generate(ref_pose, ref_pose_frame_dir, args.preprocess, source_image_flag=False)
            if ref_pose_coeff_path is None:
                print("Warning: Failed to extract coefficients from reference pose video. Proceeding without pose reference.")
    else:
        ref_pose_coeff_path=None

    #audio2ceoff
    if profile and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=args.still)
    if profile and torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    if profile:
        print(f'[profile] prepare audio/batch time: {t1-t0:.3f}s')

    if profile and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    
    # Use mixed precision for audio to coefficient generation if available
    if use_amp:
        with torch.cuda.amp.autocast():
            coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)
    else:
        coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)
    
    if profile and torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    if profile:
        device_info = f"({'CUDA+AMP' if use_amp else 'CUDA' if args.device == 'cuda' else 'CPU'})"
        print(f'[profile] audio2coeff generate {device_info} time: {t1-t0:.3f}s')
    
    if coeff_path is None:
        raise RuntimeError("Failed to generate audio coefficients. Please check the audio file format and quality.")

    # 3dface render
    if args.face3dvis:
        from src.face3d.visualize import gen_composed_video
        gen_composed_video(args, device, first_coeff_path, coeff_path, audio_path, os.path.join(save_dir, '3dface.mp4'))
    
    #coeff2video
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, 
                                batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                                expression_scale=args.expression_scale, still_mode=args.still, preprocess=args.preprocess, size=args.size)
    
    # Dynamic batch size adjustment based on GPU memory
    effective_batch_size = batch_size
    if torch.cuda.is_available() and args.device == "cuda":
        try:
            # Check available GPU memory
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated_memory_gb = torch.cuda.memory_allocated() / 1024**3
            available_memory_gb = gpu_memory_gb - allocated_memory_gb
            
            # Warn about enhancer memory usage
            if (args.enhancer or args.background_enhancer) and available_memory_gb < 6:
                print(f"Warning: Using enhancer with limited GPU memory ({available_memory_gb:.1f}GB available). Consider disabling enhancer or running on CPU if you encounter out-of-memory errors.")
            
            # Adjust batch size based on available memory
            if available_memory_gb < 4:  # Less than 4GB available
                effective_batch_size = max(1, batch_size // 4)
                print(f"GPU memory low ({available_memory_gb:.1f}GB available), reducing batch size to {effective_batch_size}")
            elif available_memory_gb < 8:  # Less than 8GB available
                effective_batch_size = max(1, batch_size // 2)
                print(f"GPU memory moderate ({available_memory_gb:.1f}GB available), reducing batch size to {effective_batch_size}")
            
        except Exception as e:
            print(f"Warning: Could not check GPU memory, using default batch size: {e}")
    
    if profile and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    
    try:
        result = animate_from_coeff.generate(data, save_dir, pic_path, crop_info, \
                                    enhancer=args.enhancer, background_enhancer=args.background_enhancer, 
                                    preprocess=args.preprocess, img_size=args.size, profile=profile, 
                                    batch_size=effective_batch_size)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and effective_batch_size > 1:
            print(f"GPU out of memory with batch size {effective_batch_size}, retrying with batch size 1...")
            torch.cuda.empty_cache()  # Clear GPU cache
            result = animate_from_coeff.generate(data, save_dir, pic_path, crop_info, \
                                        enhancer=args.enhancer, background_enhancer=args.background_enhancer, 
                                        preprocess=args.preprocess, img_size=args.size, profile=profile, 
                                        batch_size=1)
        else:
            raise e
    
    if profile and torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    if profile:
        device_info = f"({'CUDA' if args.device == 'cuda' else 'CPU'})"
        print(f'[profile] facerender total generate {device_info} time: {t1-t0:.3f}s (batch_size={effective_batch_size})')
    
    if result is None:
        raise RuntimeError("Failed to generate final video. Please check all inputs and try again.")
    
    # Use try/except for file operations to handle potential issues
    try:
        shutil.move(result, save_dir+'.mp4')
        print('The generated video is named:', save_dir+'.mp4')
    except Exception as e:
        print(f"Warning: Could not move final video file: {e}")
        print(f'The generated video is at: {result}')

    if not args.verbose:
        try:
            shutil.rmtree(save_dir)
        except Exception as e:
            print(f"Warning: Could not clean up temporary directory {save_dir}: {e}")

    # Final GPU memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if profile:
            print(f"[profile] GPU memory cleared. Final allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

    
if __name__ == '__main__':

    parser = ArgumentParser()  
    parser.add_argument("--driven_audio", default='./examples/driven_audio/bus_chinese.wav', help="path to driven audio")
    parser.add_argument("--source_image", default='./examples/source_image/full_body_1.png', help="path to source image")
    parser.add_argument("--ref_eyeblink", default=None, help="path to reference video providing eye blinking")
    parser.add_argument("--ref_pose", default=None, help="path to reference video providing pose")
    parser.add_argument("--checkpoint_dir", default='./checkpoints', help="path to output")
    parser.add_argument("--result_dir", default='./results', help="path to output")
    parser.add_argument("--pose_style", type=int, default=0,  help="input pose style from [0, 46)")
    parser.add_argument("--batch_size", type=int, default=8,  help="the batch size of facerender")
    parser.add_argument("--size", type=int, default=256,  help="the image size of the facerender")
    parser.add_argument("--expression_scale", type=float, default=1.,  help="the batch size of facerender")
    parser.add_argument('--input_yaw', nargs='+', type=int, default=None, help="the input yaw degree of the user ")
    parser.add_argument('--input_pitch', nargs='+', type=int, default=None, help="the input pitch degree of the user")
    parser.add_argument('--input_roll', nargs='+', type=int, default=None, help="the input roll degree of the user")
    parser.add_argument('--enhancer',  type=str, default=None, help="Face enhancer, [gfpgan, RestoreFormer]. Warning: Increases VRAM usage significantly")
    parser.add_argument('--background_enhancer',  type=str, default=None, help="background enhancer, [realesrgan]. Warning: Increases VRAM usage significantly")
    parser.add_argument('--profile', action='store_true', help='print timing breakdown for major stages')
    
    # NEW: Performance optimization arguments
    parser.add_argument('--optimization_preset', type=str, choices=['ultra_fast', 'fast', 'balanced', 'quality'], 
                       default='balanced', help="Performance optimization preset")
    parser.add_argument('--optimization_level', type=str, choices=['low', 'medium', 'high', 'extreme'], 
                       help="Manual optimization level (overrides preset)")
    parser.add_argument('--list_presets', action='store_true', help="List available optimization presets and exit")
    parser.add_argument("--cpu", dest="cpu", action="store_true") 
    parser.add_argument("--face3dvis", action="store_true", help="generate 3d face and 3d landmarks") 
    parser.add_argument("--still", action="store_true", help="can crop back to the original videos for the full body aniamtion") 
    parser.add_argument("--preprocess", default='crop', choices=['crop', 'extcrop', 'resize', 'full', 'extfull'], help="how to preprocess the images" ) 
    parser.add_argument("--verbose",action="store_true", help="saving the intermedia output or not" ) 
    parser.add_argument("--old_version",action="store_true", help="use the pth other than safetensor version" ) 


    # net structure and parameters (kept for compatibility, but consider removing if truly unused)
    parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='Network architecture for face reconstruction')
    parser.add_argument('--init_path', type=str, default=None, help='Path for network initialization')
    parser.add_argument('--use_last_fc',default=False, help='zero initialize the last fc')
    parser.add_argument('--bfm_folder', type=str, default='./checkpoints/BFM_Fitting/')
    parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

    # default renderer parameters
    parser.add_argument('--focal', type=float, default=1015.)
    parser.add_argument('--center', type=float, default=112.)
    parser.add_argument('--camera_d', type=float, default=10.)
    parser.add_argument('--z_near', type=float, default=5.)
    parser.add_argument('--z_far', type=float, default=15.)

    args = parser.parse_args()
    
    # Handle preset listing
    if args.list_presets:
        OptimizationConfig.list_presets()
        return

    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda"
    else:
        args.device = "cpu"

    main(args)

