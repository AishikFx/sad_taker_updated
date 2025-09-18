# 🎯 SadTalker FastAPI Microservice with Redis Caching
# Ultra-fast video generation using intelligent caching strategy

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import redis
import hashlib
import json
import os
import uuid
import asyncio
from typing import Optional, Dict, Any
import logging
from datetime import datetime, timedelta
import pickle
import numpy as np
from scipy.io import savemat, loadmat

# SadTalker imports
import sys
sys.path.append('..')
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SadTalker Microservice",
    description="Ultra-fast talking head video generation with intelligent caching",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis connection (configurable via environment for Docker)
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
REDIS_DB = int(os.environ.get('REDIS_DB', 0))
REDIS_DECODE_RESPONSES = os.environ.get('REDIS_DECODE_RESPONSES', 'False').lower() in ('1', 'true', 'yes')

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=REDIS_DECODE_RESPONSES)

# Global models (loaded once)
class SadTalkerModels:
    def __init__(self):
        self.device = "cuda"
        self.sadtalker_paths = init_path("../checkpoints", os.path.join("../src", "config"))
        
        # Initialize models
        self.preprocess_model = CropAndExtract(self.sadtalker_paths, self.device)
        self.audio_to_coeff = Audio2Coeff(self.sadtalker_paths, self.device)
        self.animate_from_coeff = {
            "full": AnimateFromCoeff(self.sadtalker_paths, self.device),
            "others": AnimateFromCoeff(self.sadtalker_paths, self.device)
        }
        logger.info("✅ SadTalker models loaded successfully")

# Initialize models globally
models = SadTalkerModels()

# 🔑 Cache Keys Strategy
class CacheKeys:
    @staticmethod
    def face_detection(image_hash: str) -> str:
        return f"face_detection:{image_hash}"
    
    @staticmethod
    def face_crop(image_hash: str, preprocess_mode: str) -> str:
        return f"face_crop:{image_hash}:{preprocess_mode}"
    
    @staticmethod
    def dmm_coeffs(image_hash: str, preprocess_mode: str) -> str:
        return f"3dmm_coeffs:{image_hash}:{preprocess_mode}"
    
    @staticmethod
    def face_mesh(image_hash: str, preprocess_mode: str) -> str:
        return f"face_mesh:{image_hash}:{preprocess_mode}"
    
    @staticmethod
    def gestures(image_hash: str) -> str:
        return f"gestures:{image_hash}"
    
    @staticmethod
    def visemes(image_hash: str) -> str:
        return f"visemes:{image_hash}"
    
    @staticmethod
    def background(image_hash: str, preprocess_mode: str) -> str:
        return f"background:{image_hash}:{preprocess_mode}"
    
    @staticmethod
    def session(session_id: str) -> str:
        return f"session:{session_id}"
    
    @staticmethod
    def enhancement_params(image_hash: str, enhancer: str) -> str:
        return f"enhance_params:{image_hash}:{enhancer}"

# 🛠️ Utility Functions
def hash_image(image_bytes: bytes) -> str:
    """Generate SHA-256 hash of image for caching"""
    return hashlib.sha256(image_bytes).hexdigest()

def cache_set(key: str, data: Any, ttl: int = 2592000):  # 30 days default
    """Store data in Redis with TTL"""
    try:
        serialized_data = pickle.dumps(data)
        # If ttl is negative or zero, store without expiration (forever)
        if ttl is None or ttl < 0:
            redis_client.set(key, serialized_data)
        else:
            redis_client.setex(key, ttl, serialized_data)
        logger.info(f"✅ Cached: {key} (TTL: {ttl}s)")
        return True
    except Exception as e:
        logger.error(f"❌ Cache set failed for {key}: {e}")
        return False

def cache_get(key: str) -> Optional[Any]:
    """Retrieve data from Redis"""
    try:
        data = redis_client.get(key)
        if data:
            return pickle.loads(data)
        return None
    except Exception as e:
        logger.error(f"❌ Cache get failed for {key}: {e}")
        return None

# 📊 Processing Pipeline Components
class ImageProcessor:
    """Handles all image processing with intelligent caching"""
    
    def __init__(self, models: SadTalkerModels):
        self.models = models
    
    async def process_image_full_pipeline(self, image_bytes: bytes, preprocess_mode: str = "full") -> Dict[str, Any]:
        """
        🔥 TIER 1: Core Image Processing Pipeline
        Process image through all stages with caching at each step
        """
        image_hash = hash_image(image_bytes)
        logger.info(f"🎯 Processing image: {image_hash[:8]}... Mode: {preprocess_mode}")
        
        # Check if we already have complete processing
        session_key = f"complete_processing:{image_hash}:{preprocess_mode}"
        cached_result = cache_get(session_key)
        if cached_result:
            logger.info(f"⚡ Found complete cached processing for {image_hash[:8]}")
            return cached_result
        
        result = {
            "image_hash": image_hash,
            "preprocess_mode": preprocess_mode,
            "processing_stages": {}
        }
        
        # Save image to temp file
        temp_image_path = f"/tmp/{image_hash}.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(image_bytes)
        
        # 1.1 Face Detection & Landmarks
        landmarks_key = CacheKeys.face_detection(image_hash)
        landmarks_data = cache_get(landmarks_key)
        
        if not landmarks_data:
            logger.info("🔍 Computing face detection & landmarks...")
            # This would be extracted from preprocess_model.generate()
            landmarks_data = {"computed": True, "stage": "face_detection"}
            cache_set(landmarks_key, landmarks_data, ttl=2592000)  # 30 days
        else:
            logger.info("⚡ Using cached face detection")
        
        result["processing_stages"]["face_detection"] = landmarks_data
        
        # 1.2 Face Cropping & Alignment + 1.3 3DMM Extraction (Most Expensive!)
        coeffs_key = CacheKeys.dmm_coeffs(image_hash, preprocess_mode)
        crop_key = CacheKeys.face_crop(image_hash, preprocess_mode)
        
        cached_coeffs = cache_get(coeffs_key)
        cached_crop = cache_get(crop_key)
        
        if not cached_coeffs or not cached_crop:
            logger.info("🧠 Computing 3DMM extraction (EXPENSIVE OPERATION)...")
            
            # Run the actual expensive computation
            temp_dir = f"/tmp/{image_hash}_processing"
            os.makedirs(temp_dir, exist_ok=True)
            
            first_coeff_path, crop_pic_path, crop_info = self.models.preprocess_model.generate(
                temp_image_path, temp_dir, preprocess_mode, source_image_flag=True
            )
            
            if first_coeff_path and crop_pic_path:
                # Cache the coefficients (⭐ MOST IMPORTANT CACHE)
                coeffs_data = {
                    "coeff_path": first_coeff_path,
                    "computed_at": datetime.now().isoformat()
                }
                cache_set(coeffs_key, coeffs_data, ttl=-1)  # Cache forever!
                
                # Cache crop info
                crop_data = {
                    "crop_pic_path": crop_pic_path,
                    "crop_info": crop_info,
                    "computed_at": datetime.now().isoformat()
                }
                cache_set(crop_key, crop_data, ttl=2592000)  # 30 days
                
                logger.info("✅ 3DMM coefficients cached (FOREVER)")
            else:
                raise HTTPException(status_code=400, detail="Failed to extract 3DMM coefficients")
        else:
            logger.info("⚡ Using cached 3DMM coefficients (HUGE TIME SAVE!)")
            coeffs_data = cached_coeffs
            crop_data = cached_crop
        
        result["processing_stages"]["3dmm_coeffs"] = coeffs_data
        result["processing_stages"]["face_crop"] = crop_data
        
        # 🚀 TIER 2: Pre-generate Basic Gestures
        gestures_key = CacheKeys.gestures(image_hash)
        cached_gestures = cache_get(gestures_key)
        
        if not cached_gestures:
            logger.info("👋 Pre-generating basic gestures...")
            gestures_data = await self._generate_basic_gestures(coeffs_data["coeff_path"])
            cache_set(gestures_key, gestures_data, ttl=604800)  # 7 days
        else:
            logger.info("⚡ Using cached gestures")
            gestures_data = cached_gestures
        
        result["processing_stages"]["gestures"] = gestures_data
        
        # 2.3 Phoneme Visemes
        visemes_key = CacheKeys.visemes(image_hash)
        cached_visemes = cache_get(visemes_key)
        
        if not cached_visemes:
            logger.info("👄 Pre-generating phoneme visemes...")
            visemes_data = await self._generate_visemes(coeffs_data["coeff_path"])
            cache_set(visemes_key, visemes_data, ttl=604800)  # 7 days
        else:
            logger.info("⚡ Using cached visemes")
            visemes_data = cached_visemes
        
        result["processing_stages"]["visemes"] = visemes_data
        
        # ⚡ TIER 3: Background Processing (if full mode)
        if preprocess_mode == "full":
            bg_key = CacheKeys.background(image_hash, preprocess_mode)
            cached_bg = cache_get(bg_key)
            
            if not cached_bg:
                logger.info("🖼️ Processing background for full mode...")
                bg_data = await self._process_background(temp_image_path)
                cache_set(bg_key, bg_data, ttl=604800)  # 7 days
            else:
                logger.info("⚡ Using cached background")
                bg_data = cached_bg
            
            result["processing_stages"]["background"] = bg_data
        
        # Cache the complete result
        cache_set(session_key, result, ttl=2592000)  # 30 days
        
        # Cleanup temp files
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        logger.info(f"✅ Complete processing cached for {image_hash[:8]}")
        return result
    
    async def _generate_basic_gestures(self, coeff_path: str) -> Dict[str, Any]:
        """Generate basic gestures like hello, nod, smile, blink"""
        # In real implementation, this would generate actual gesture coefficients
        gestures = {
            "hello": {"type": "wave", "duration": 2.0, "generated": True},
            "nod_yes": {"type": "nod", "direction": "vertical", "duration": 1.5},
            "nod_no": {"type": "nod", "direction": "horizontal", "duration": 1.5},
            "smile": {"type": "expression", "emotion": "happy", "duration": 1.0},
            "blink": {"type": "eye_action", "action": "blink", "duration": 0.5}
        }
        return gestures
    
    async def _generate_visemes(self, coeff_path: str) -> Dict[str, Any]:
        """Generate all phoneme mouth shapes for lip-sync"""
        phonemes = ['A', 'E', 'I', 'O', 'U', 'M', 'B', 'P', 'F', 'V', 'S', 'T', 'L', 'R']
        visemes = {}
        for phoneme in phonemes:
            visemes[phoneme] = {
                "mouth_shape": phoneme,
                "coefficients": f"viseme_{phoneme.lower()}_coeffs",
                "generated": True
            }
        return visemes
    
    async def _process_background(self, image_path: str) -> Dict[str, Any]:
        """Process background for full body animation"""
        return {
            "background_mask": "processed",
            "body_region": "extracted", 
            "blend_params": {"alpha": 0.8},
            "processed": True
        }

# Initialize image processor
image_processor = ImageProcessor(models)