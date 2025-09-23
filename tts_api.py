"""
SadTalker TTS API Endpoints
Provides text-to-speech video generation capabilities.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import tempfile
import logging
from typing import Optional
from pydantic import BaseModel

# Import the TTS integration and SadTalker inference
from src.utils.tts_integration import SadTalkerTTS
import sys
sys.path.append('..')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
tts_app = FastAPI(
    title="SadTalker TTS API",
    description="Text-to-Speech video generation using SadTalker",
    version="1.0.0"
)

# CORS middleware
tts_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class TTSVideoRequest(BaseModel):
    text: str
    gender: str = "female"
    enhancer: Optional[str] = None
    background_enhancer: Optional[str] = None
    size: int = 256
    expression_scale: float = 1.0
    pose_style: int = 0

class TTSVideoResponse(BaseModel):
    success: bool
    message: str
    video_path: Optional[str] = None
    session_id: str
    processing_time: Optional[float] = None

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    details: Optional[str] = None


@tts_app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "SadTalker TTS API",
        "version": "1.0.0",
        "description": "Generate talking head videos from text and images",
        "supported_genders": ["female"],
        "endpoints": {
            "/generate-video": "POST - Generate video from text and image",
            "/health": "GET - Health check",
            "/supported-genders": "GET - List supported voice genders"
        }
    }

@tts_app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Quick test of TTS availability
        tts = SadTalkerTTS()
        return {
            "status": "healthy",
            "tts_available": True,
            "supported_genders": tts.supported_genders
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "tts_available": False,
            "error": str(e)
        }

@tts_app.get("/supported-genders")
async def get_supported_genders():
    """Get list of supported voice genders."""
    try:
        tts = SadTalkerTTS()
        return {
            "supported_genders": tts.supported_genders,
            "default": "female",
            "note": "Currently only female voice with Indian accent is supported"
        }
    except Exception as e:
        return {"error": str(e)}

@tts_app.post("/generate-video", response_model=TTSVideoResponse)
async def generate_video_from_text(
    image: UploadFile = File(..., description="Source image file"),
    text: str = Form(..., description="Text to convert to speech"),
    gender: str = Form("female", description="Voice gender (only 'female' supported)"),
    enhancer: Optional[str] = Form(None, description="Face enhancer (gfpgan, RestoreFormer)"),
    background_enhancer: Optional[str] = Form(None, description="Background enhancer (realesrgan)"),
    size: int = Form(256, description="Output image size"),
    expression_scale: float = Form(1.0, description="Expression scale factor"),
    pose_style: int = Form(0, description="Pose style (0-46)")
):
    """
    Generate a talking head video from text and image.
    
    This endpoint:
    1. Accepts an image file and text input
    2. Converts text to speech using Google TTS (Indian accent for female voice)
    3. Generates a video using SadTalker with the synthesized audio
    4. Returns the generated video file
    """
    session_id = str(uuid.uuid4())
    
    try:
        import time
        start_time = time.time()
        
        logger.info(f"🎬 Starting TTS video generation - Session: {session_id}")
        logger.info(f"📝 Text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        logger.info(f"👤 Gender: {gender}")
        
        # Validate inputs
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if gender.lower() != 'female':
            raise HTTPException(
                status_code=400, 
                detail="Only 'female' gender is currently supported for TTS"
            )
        
        # Validate image file
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Create temporary directories
        temp_dir = tempfile.mkdtemp(prefix=f"sadtalker_tts_{session_id}_")
        logger.info(f"📁 Working directory: {temp_dir}")
        
        # Save uploaded image
        image_path = os.path.join(temp_dir, f"source_image_{session_id}.{image.filename.split('.')[-1]}")
        with open(image_path, "wb") as f:
            content = await image.read()
            f.write(content)
        
        logger.info(f"💾 Image saved: {image_path}")
        
        # Generate TTS audio
        logger.info("🎤 Generating speech from text...")
        tts = SadTalkerTTS()
        audio_path = tts.text_to_audio_for_sadtalker(text, gender)
        logger.info(f"🎵 Audio generated: {audio_path}")
        
        # Prepare SadTalker arguments
        class Args:
            def __init__(self):
                self.source_image = image_path
                self.driven_audio = audio_path
                self.input_text = text
                self.gender = gender
                self.result_dir = temp_dir
                self.checkpoint_dir = './checkpoints'
                self.size = size
                self.expression_scale = expression_scale
                self.pose_style = pose_style
                self.enhancer = enhancer
                self.background_enhancer = background_enhancer
                self.device = "cuda" if os.environ.get('CUDA_AVAILABLE', 'true').lower() == 'true' else "cpu"
                self.batch_size = 4  # Conservative batch size
                self.optimization_preset = 'balanced'
                self.preprocess = 'crop'
                self.still = False
                self.face3dvis = False
                self.old_version = False
                self.verbose = False
                self.cpu = False
                self.profile = False
                
                # Optional parameters (set to None if not provided)
                self.ref_eyeblink = None
                self.ref_pose = None
                self.input_yaw = None
                self.input_pitch = None
                self.input_roll = None
                
                # Advanced parameters with defaults
                self.net_recon = 'resnet50'
                self.init_path = None
                self.use_last_fc = False
                self.bfm_folder = './checkpoints/BFM_Fitting/'
                self.bfm_model = 'BFM_model_front.mat'
                self.focal = 1015.0
                self.center = 112.0
                self.camera_d = 10.0
                self.z_near = 5.0
                self.z_far = 15.0
        
        args = Args()
        
        # Import and run SadTalker inference
        logger.info("🎭 Starting SadTalker video generation...")
        from inference import main as sadtalker_main
        
        try:
            output_video_path = sadtalker_main(args)
            processing_time = time.time() - start_time
            
            if output_video_path and os.path.exists(output_video_path):
                logger.info(f"✅ Video generated successfully: {output_video_path}")
                logger.info(f"⏱️ Total processing time: {processing_time:.2f}s")
                
                return TTSVideoResponse(
                    success=True,
                    message="Video generated successfully from text",
                    video_path=output_video_path,
                    session_id=session_id,
                    processing_time=processing_time
                )
            else:
                raise RuntimeError("Video generation completed but output file not found")
                
        except Exception as e:
            logger.error(f"❌ SadTalker generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Video generation failed: {str(e)}")
        
        finally:
            # Cleanup temporary audio file
            try:
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
                    logger.info(f"🧹 Cleaned up TTS audio: {audio_path}")
            except Exception as e:
                logger.warning(f"⚠️ Could not cleanup TTS audio: {e}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in TTS video generation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@tts_app.get("/download/{session_id}")
async def download_video(session_id: str):
    """Download generated video by session ID."""
    # This is a simplified version - in production you'd want to store
    # session information and video paths in a database or cache
    
    # For now, return an error as we don't have session storage implemented
    raise HTTPException(
        status_code=501, 
        detail="Video download by session ID not implemented yet. Videos are returned directly from /generate-video endpoint."
    )

@tts_app.post("/generate-video-simple")
async def generate_video_simple(
    image: UploadFile = File(...),
    text: str = Form(...),
    gender: str = Form("female")
):
    """
    Simplified endpoint for basic video generation with minimal parameters.
    """
    return await generate_video_from_text(
        image=image,
        text=text,
        gender=gender,
        enhancer=None,
        background_enhancer=None,
        size=256,
        expression_scale=1.0,
        pose_style=0
    )


# Error handlers
@tts_app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return {
        "success": False,
        "error": exc.detail,
        "status_code": exc.status_code
    }

@tts_app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return {
        "success": False,
        "error": "Internal server error",
        "details": str(exc)
    }


# Startup event
@tts_app.on_event("startup")
async def startup_event():
    """Initialize TTS system on startup."""
    logger.info("🚀 SadTalker TTS API starting up...")
    
    try:
        # Test TTS initialization
        tts = SadTalkerTTS()
        logger.info("✅ TTS system initialized successfully")
        logger.info(f"🎭 Supported genders: {tts.supported_genders}")
        
    except Exception as e:
        logger.error(f"❌ TTS initialization failed: {e}")
        logger.error("Please ensure gtts and pydub are installed: pip install gtts pydub")

# Main app instance for importing
app = tts_app

if __name__ == "__main__":
    import uvicorn
    
    print("🎬 Starting SadTalker TTS API Server...")
    print("📖 API Documentation will be available at: http://localhost:8000/docs")
    print("🎭 Endpoints:")
    print("   POST /generate-video - Generate video from text and image")
    print("   POST /generate-video-simple - Simplified video generation")
    print("   GET /health - Health check")
    print("   GET /supported-genders - List supported voice genders")
    
    uvicorn.run(
        "tts_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )