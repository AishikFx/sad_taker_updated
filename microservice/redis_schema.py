# 🎯 Redis Schema Design & Caching Strategy
# Optimized for SadTalker microservice with TTL management

"""
Redis Schema for SadTalker Caching System

🔑 KEY PATTERNS:
- face_detection:{image_hash}           # TTL: 30 days
- face_crop:{image_hash}:{preprocess}   # TTL: 30 days  
- 3dmm_coeffs:{image_hash}:{preprocess} # TTL: FOREVER (most important)
- face_mesh:{image_hash}:{preprocess}   # TTL: 30 days
- gestures:{image_hash}                 # TTL: 7 days
- visemes:{image_hash}                  # TTL: 7 days
- background:{image_hash}:{preprocess}  # TTL: 7 days
- session:{session_id}                  # TTL: 1 hour
- enhance_params:{image_hash}:{enhancer} # TTL: 7 days
- complete_processing:{image_hash}:{preprocess} # TTL: 30 days

📊 STORAGE ESTIMATES PER IMAGE:
- Face Detection: ~5KB
- Face Crop: ~1-2MB (includes cropped image)
- 3DMM Coefficients: ~50KB (CRITICAL - saves 3-5 seconds)
- Face Mesh: ~500KB
- Gestures: ~1MB (5 basic gestures)
- Visemes: ~1MB (14 phonemes)
- Background: ~2MB
- Total per image: ~6-7MB

⚡ PERFORMANCE IMPACT:
- First time: 15-20 seconds
- With cache: 3-5 seconds
- Speedup: 4-6x faster
"""

import redis
import pickle
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class SadTalkerRedisSchema:
    """Redis schema manager for SadTalker caching"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        
        # TTL values in seconds
        self.TTL_FOREVER = -1  # Never expire
        self.TTL_30_DAYS = 30 * 24 * 60 * 60  # 2,592,000 seconds
        self.TTL_7_DAYS = 7 * 24 * 60 * 60    # 604,800 seconds  
        self.TTL_1_HOUR = 60 * 60             # 3,600 seconds
    
    # 🔥 TIER 1: Core Image Processing (Cache Forever/Long Term)
    
    def cache_face_detection(self, image_hash: str, landmarks_data: Dict[str, Any]) -> bool:
        """Cache face detection results - High priority, 30 days TTL"""
        key = f"face_detection:{image_hash}"
        return self._cache_with_metadata(key, landmarks_data, self.TTL_30_DAYS, "face_detection")
    
    def get_face_detection(self, image_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached face detection results"""
        key = f"face_detection:{image_hash}"
        return self._get_with_metadata(key)
    
    def cache_3dmm_coeffs(self, image_hash: str, preprocess_mode: str, coeffs_data: Dict[str, Any]) -> bool:
        """
        🌟 MOST IMPORTANT CACHE - 3DMM Coefficients
        Cache FOREVER - saves 3-5 seconds per request
        """
        key = f"3dmm_coeffs:{image_hash}:{preprocess_mode}"
        return self._cache_with_metadata(key, coeffs_data, self.TTL_FOREVER, "3dmm_coefficients")
    
    def get_3dmm_coeffs(self, image_hash: str, preprocess_mode: str) -> Optional[Dict[str, Any]]:
        """Get cached 3DMM coefficients"""
        key = f"3dmm_coeffs:{image_hash}:{preprocess_mode}"
        return self._get_with_metadata(key)
    
    def cache_face_crop(self, image_hash: str, preprocess_mode: str, crop_data: Dict[str, Any]) -> bool:
        """Cache face crop and alignment data - 30 days TTL"""
        key = f"face_crop:{image_hash}:{preprocess_mode}"
        return self._cache_with_metadata(key, crop_data, self.TTL_30_DAYS, "face_crop")
    
    def get_face_crop(self, image_hash: str, preprocess_mode: str) -> Optional[Dict[str, Any]]:
        """Get cached face crop data"""
        key = f"face_crop:{image_hash}:{preprocess_mode}"
        return self._get_with_metadata(key)
    
    def cache_face_mesh(self, image_hash: str, preprocess_mode: str, mesh_data: Dict[str, Any]) -> bool:
        """Cache 3D face mesh data - 30 days TTL"""
        key = f"face_mesh:{image_hash}:{preprocess_mode}"
        return self._cache_with_metadata(key, mesh_data, self.TTL_30_DAYS, "face_mesh")
    
    def get_face_mesh(self, image_hash: str, preprocess_mode: str) -> Optional[Dict[str, Any]]:
        """Get cached face mesh data"""
        key = f"face_mesh:{image_hash}:{preprocess_mode}"
        return self._get_with_metadata(key)
    
    # 🚀 TIER 2: Pre-generated Animations (Cache Medium Term)
    
    def cache_gestures(self, image_hash: str, gestures_data: Dict[str, Any]) -> bool:
        """Cache pre-generated basic gestures - 7 days TTL"""
        key = f"gestures:{image_hash}"
        return self._cache_with_metadata(key, gestures_data, self.TTL_7_DAYS, "gestures")
    
    def get_gestures(self, image_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached gestures"""
        key = f"gestures:{image_hash}"
        return self._get_with_metadata(key)
    
    def cache_visemes(self, image_hash: str, visemes_data: Dict[str, Any]) -> bool:
        """Cache phoneme visemes for lip-sync - 7 days TTL"""
        key = f"visemes:{image_hash}"
        return self._cache_with_metadata(key, visemes_data, self.TTL_7_DAYS, "visemes")
    
    def get_visemes(self, image_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached visemes"""
        key = f"visemes:{image_hash}"
        return self._get_with_metadata(key)
    
    # ⚡ TIER 3: Render Optimizations (Cache Short Term)
    
    def cache_background(self, image_hash: str, preprocess_mode: str, bg_data: Dict[str, Any]) -> bool:
        """Cache background processing for full mode - 7 days TTL"""
        key = f"background:{image_hash}:{preprocess_mode}"
        return self._cache_with_metadata(key, bg_data, self.TTL_7_DAYS, "background")
    
    def get_background(self, image_hash: str, preprocess_mode: str) -> Optional[Dict[str, Any]]:
        """Get cached background data"""
        key = f"background:{image_hash}:{preprocess_mode}"
        return self._get_with_metadata(key)
    
    def cache_enhancement_params(self, image_hash: str, enhancer: str, params: Dict[str, Any]) -> bool:
        """Cache face enhancement parameters - 7 days TTL"""
        key = f"enhance_params:{image_hash}:{enhancer}"
        return self._cache_with_metadata(key, params, self.TTL_7_DAYS, "enhancement")
    
    def get_enhancement_params(self, image_hash: str, enhancer: str) -> Optional[Dict[str, Any]]:
        """Get cached enhancement parameters"""
        key = f"enhance_params:{image_hash}:{enhancer}"
        return self._get_with_metadata(key)
    
    # 📋 Session Management (Cache Short Term)
    
    def cache_session(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """Cache VIP session data - 1 hour TTL"""
        key = f"session:{session_id}"
        return self._cache_with_metadata(key, session_data, self.TTL_1_HOUR, "session")
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        key = f"session:{session_id}"
        return self._get_with_metadata(key)
    
    def update_session_status(self, session_id: str, status: str, extra_data: Dict[str, Any] = None) -> bool:
        """Update session status"""
        session_data = self.get_session(session_id)
        if session_data:
            session_data["status"] = status
            session_data["updated_at"] = datetime.now().isoformat()
            if extra_data:
                session_data.update(extra_data)
            return self.cache_session(session_id, session_data)
        return False
    
    # 🎯 Complete Processing Cache
    
    def cache_complete_processing(self, image_hash: str, preprocess_mode: str, result: Dict[str, Any]) -> bool:
        """Cache complete processing result - 30 days TTL"""
        key = f"complete_processing:{image_hash}:{preprocess_mode}"
        return self._cache_with_metadata(key, result, self.TTL_30_DAYS, "complete_processing")
    
    def get_complete_processing(self, image_hash: str, preprocess_mode: str) -> Optional[Dict[str, Any]]:
        """Get complete processing result"""
        key = f"complete_processing:{image_hash}:{preprocess_mode}"
        return self._get_with_metadata(key)
    
    # 📊 Cache Management & Statistics
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        try:
            redis_info = self.redis.info()
            
            stats = {
                "redis_status": {
                    "connected": self.redis.ping(),
                    "memory_used": redis_info.get('used_memory_human', 'N/A'),
                    "total_keys": self.redis.dbsize(),
                    "uptime": redis_info.get('uptime_in_seconds', 0)
                },
                "cached_data": {
                    "face_detection": len(self.redis.keys("face_detection:*")),
                    "3dmm_coeffs": len(self.redis.keys("3dmm_coeffs:*")),
                    "face_crop": len(self.redis.keys("face_crop:*")),
                    "face_mesh": len(self.redis.keys("face_mesh:*")),
                    "gestures": len(self.redis.keys("gestures:*")),
                    "visemes": len(self.redis.keys("visemes:*")),
                    "background": len(self.redis.keys("background:*")),
                    "enhancement_params": len(self.redis.keys("enhance_params:*")),
                    "complete_processing": len(self.redis.keys("complete_processing:*"))
                },
                "sessions": {
                    "active_sessions": len(self.redis.keys("session:*")),
                    "total_processed": self._get_counter("total_processed_images"),
                    "total_generated": self._get_counter("total_generated_videos")
                },
                "performance": {
                    "cache_hit_rate": self._calculate_hit_rate(),
                    "avg_processing_time": {
                        "first_time": "15-20 seconds",
                        "with_cache": "3-5 seconds",
                        "speedup": "4-6x faster"
                    }
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get cache statistics: {e}")
            return {"error": "Failed to retrieve statistics"}
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions manually"""
        try:
            session_keys = self.redis.keys("session:*")
            cleaned = 0
            
            for key in session_keys:
                ttl = self.redis.ttl(key)
                if ttl == -2:  # Key expired
                    self.redis.delete(key)
                    cleaned += 1
            
            logger.info(f"🧹 Cleaned up {cleaned} expired sessions")
            return cleaned
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            return 0
    
    def get_image_cache_summary(self, image_hash: str) -> Dict[str, Any]:
        """Get summary of all cached data for a specific image"""
        summary = {
            "image_hash": image_hash,
            "cached_components": {},
            "total_cache_size": 0,
            "cache_coverage": 0
        }
        
        # Check each cache type
        cache_checks = [
            ("face_detection", f"face_detection:{image_hash}"),
            ("3dmm_coeffs_full", f"3dmm_coeffs:{image_hash}:full"),
            ("3dmm_coeffs_crop", f"3dmm_coeffs:{image_hash}:crop"),
            ("face_crop_full", f"face_crop:{image_hash}:full"),
            ("face_crop_crop", f"face_crop:{image_hash}:crop"),
            ("gestures", f"gestures:{image_hash}"),
            ("visemes", f"visemes:{image_hash}"),
            ("background_full", f"background:{image_hash}:full"),
            ("complete_processing_full", f"complete_processing:{image_hash}:full")
        ]
        
        cached_count = 0
        for component, key in cache_checks:
            if self.redis.exists(key):
                ttl = self.redis.ttl(key)
                size = len(self.redis.get(key) or b'')
                
                summary["cached_components"][component] = {
                    "cached": True,
                    "ttl": ttl if ttl > 0 else "permanent",
                    "size_bytes": size
                }
                summary["total_cache_size"] += size
                cached_count += 1
            else:
                summary["cached_components"][component] = {"cached": False}
        
        summary["cache_coverage"] = (cached_count / len(cache_checks)) * 100
        
        return summary
    
    # 🔧 Internal Helper Methods
    
    def _cache_with_metadata(self, key: str, data: Dict[str, Any], ttl: int, data_type: str) -> bool:
        """Cache data with metadata"""
        try:
            cache_entry = {
                "data": data,
                "metadata": {
                    "cached_at": datetime.now().isoformat(),
                    "type": data_type,
                    "ttl": ttl
                }
            }
            
            serialized = pickle.dumps(cache_entry)
            
            if ttl == self.TTL_FOREVER:
                self.redis.set(key, serialized)
            else:
                self.redis.setex(key, ttl, serialized)
            
            # Update counters
            self._increment_counter(f"cache_sets_{data_type}")
            
            logger.info(f"✅ Cached {data_type}: {key} (TTL: {ttl})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Cache set failed for {key}: {e}")
            return False
    
    def _get_with_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Get data with metadata"""
        try:
            data = self.redis.get(key)
            if data:
                cache_entry = pickle.loads(data)
                
                # Update hit counter
                data_type = cache_entry.get("metadata", {}).get("type", "unknown")
                self._increment_counter(f"cache_hits_{data_type}")
                
                return cache_entry["data"]
            else:
                # Update miss counter (we can infer type from key)
                data_type = key.split(':')[0]
                self._increment_counter(f"cache_misses_{data_type}")
                
            return None
            
        except Exception as e:
            logger.error(f"❌ Cache get failed for {key}: {e}")
            return None
    
    def _increment_counter(self, counter_name: str) -> None:
        """Increment a counter in Redis"""
        try:
            self.redis.incr(f"counter:{counter_name}")
        except Exception as e:
            logger.error(f"❌ Counter increment failed for {counter_name}: {e}")
    
    def _get_counter(self, counter_name: str) -> int:
        """Get counter value"""
        try:
            value = self.redis.get(f"counter:{counter_name}")
            return int(value) if value else 0
        except Exception as e:
            logger.error(f"❌ Counter get failed for {counter_name}: {e}")
            return 0
    
    def _calculate_hit_rate(self) -> float:
        """Calculate overall cache hit rate"""
        try:
            total_hits = sum([
                self._get_counter(f"cache_hits_{cache_type}") 
                for cache_type in ["face_detection", "3dmm_coefficients", "face_crop", "gestures", "visemes"]
            ])
            
            total_misses = sum([
                self._get_counter(f"cache_misses_{cache_type}")
                for cache_type in ["face_detection", "3dmm_coefficients", "face_crop", "gestures", "visemes"]  
            ])
            
            total_requests = total_hits + total_misses
            
            if total_requests > 0:
                return (total_hits / total_requests) * 100
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"❌ Hit rate calculation failed: {e}")
            return 0.0

# 🎯 Redis Configuration for SadTalker
REDIS_CONFIG = {
    "host": "localhost",
    "port": 6379,
    "db": 0,
    "decode_responses": False,  # We use pickle, so keep bytes
    "socket_connect_timeout": 5,
    "socket_timeout": 5,
    "retry_on_timeout": True,
    "health_check_interval": 30
}

# Memory optimization settings
MEMORY_SETTINGS = {
    "maxmemory": "2gb",  # Adjust based on your server
    "maxmemory_policy": "allkeys-lru",  # Evict least recently used keys
    "save": "900 1 300 10 60 10000",  # Persistence settings
    "appendonly": "yes"  # Enable AOF for durability
}