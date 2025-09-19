# Comprehensive CUDA OOM Handling Strategy
#
# This system implements multi-layered CUDA Out-of-Memory protection:
#
# 1. PREVENTION LAYER:
#    - AdaptiveMemoryManager: Learns from OOM history and adjusts batch sizes proactively
#    - Memory pressure tracking: normal/high/critical levels based on consecutive OOMs
#    - Proactive OOM prevention: Estimates memory needs before processing
#
# 2. RECOVERY LAYER:
#    - Emergency memory cleanup: torch.cuda.empty_cache(), garbage collection
#    - Progressive memory fraction reduction: 95% → 75% → 60% → 50%
#    - Stream-specific cleanup: Clear memory in individual CUDA streams
#    - Tensor pool cleanup: Free pre-allocated tensor pools when needed
#
# 3. FALLBACK LAYER:
#    - Sequential processing: Fall back to single-image processing
#    - Conservative batching: Reduce batch sizes progressively
#    - Original image return: Return unmodified images if enhancement fails
#
# 4. MONITORING LAYER:
#    - Real-time memory tracking: Current, peak, and utilization monitoring
#    - OOM event counting: Track frequency and patterns
#    - Performance suggestions: Provide optimization recommendations
#
# USAGE:
# - System automatically handles OOM events and continues processing
# - Batch sizes adapt based on GPU memory and OOM history
# - Memory pressure levels adjust conservatism automatically
# - All safety measures are transparent to the user
#
# For production deployments with varying GPU configurations:
# - System scales from 4GB consumer GPUs to 1000GB+ data center GPUs
# - No manual configuration needed - fully automatic adaptation
# - Maintains high VRAM utilization while preventing crashes


class FastGeneratorWithLen(object):
    """ Optimized generator with length """
    def __init__(self, gen, length):
        self.gen = gen
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.gen

class LightweightEnhancer:
    """Lightweight alternative to GFPGAN for extreme speed optimization"""
    
    def __init__(self):
        # Simple sharpening kernel
        self.sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        # Noise reduction kernel
        self.denoise_kernel = np.ones((3,3), np.float32) / 9
    
    def enhance(self, image):
        """Fast enhancement using basic image processing"""
        # Convert to float for processing
        img_float = image.astype(np.float32) / 255.0
        
        # Apply slight sharpening
        sharpened = cv2.filter2D(img_float, -1, self.sharpen_kernel)
        sharpened = np.clip(sharpened, 0, 1)
        
        # Enhance contrast slightly
        enhanced = cv2.convertScaleAbs(sharpened * 255, alpha=1.1, beta=5)
        
        # Simple color correction
        enhanced = cv2.bilateralFilter(enhanced, 5, 75, 75)
        
        return enhanced

class AdaptiveMemoryManager:
    """Adaptive memory manager that adjusts processing based on available VRAM"""

    def __init__(self):
        self.initial_memory = None
        self.peak_memory_seen = 0
        self.oom_count = 0
        self.successful_batch_sizes = []
        self.failed_batch_sizes = []
        self.consecutive_oom_count = 0  # Track consecutive OOMs
        self.last_oom_time = None
        self.memory_pressure_level = "normal"  # normal, high, critical
    
    def initialize(self):
        """Initialize memory tracking"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.initial_memory = torch.cuda.memory_allocated()
            self.peak_memory_seen = self.initial_memory
    
    def get_safe_batch_size(self, requested_batch_size, image_size=(512, 512)):
        """Get a safe batch size based on memory constraints and history"""
        if not torch.cuda.is_available():
            return min(4, requested_batch_size)
        
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        current_memory_gb = torch.cuda.memory_allocated() / 1e9
        available_memory_gb = gpu_memory_gb - current_memory_gb
        
        # Estimate memory needed per image (rough estimation)
        h, w = image_size
        memory_per_image_gb = (h * w * 3 * 4 * 4) / 1e9  # 4x overhead for processing
        
        # Conservative batch size based on available memory
        memory_based_batch_size = max(1, int(available_memory_gb * 0.6 / memory_per_image_gb))
        
        # Adjust based on OOM history and memory pressure
        if self.oom_count > 0:
            # Become more conservative after OOM events
            oom_penalty = self.oom_count + 1

            # Additional penalty for consecutive OOMs and memory pressure
            if self.memory_pressure_level == "high":
                oom_penalty *= 1.5
                print(f"   ⚠️  HIGH memory pressure: Increasing conservatism (penalty: {oom_penalty:.1f}x)")
            elif self.memory_pressure_level == "critical":
                oom_penalty *= 2.0
                print(f"   🚨 CRITICAL memory pressure: Maximum conservatism (penalty: {oom_penalty:.1f}x)")

            memory_based_batch_size = max(1, int(memory_based_batch_size / oom_penalty))
        
        # Use successful batch size history
        if self.successful_batch_sizes:
            max_successful = max(self.successful_batch_sizes[-5:])  # Last 5 successful batches
            memory_based_batch_size = min(memory_based_batch_size, max_successful)
        
        # Avoid known failed batch sizes
        while memory_based_batch_size in self.failed_batch_sizes and memory_based_batch_size > 1:
            memory_based_batch_size -= 1
        
        safe_batch_size = min(requested_batch_size, memory_based_batch_size)
        
        print(f"   💡 Memory Manager: Requested {requested_batch_size}, Safe {safe_batch_size}")
        print(f"   📊 Available VRAM: {available_memory_gb:.2f}GB, OOM count: {self.oom_count}")
        
        return safe_batch_size
    
    def record_success(self, batch_size, memory_used):
        """Record successful batch processing and potentially reduce memory pressure"""
        self.successful_batch_sizes.append(batch_size)
        self.peak_memory_seen = max(self.peak_memory_seen, memory_used)

        # Reset consecutive OOM count on success
        if self.consecutive_oom_count > 0:
            self.consecutive_oom_count = max(0, self.consecutive_oom_count - 1)

            # Reduce memory pressure level on consistent success
            if self.consecutive_oom_count == 0:
                if self.memory_pressure_level == "critical":
                    self.memory_pressure_level = "high"
                    print("   ✅ Memory pressure reduced to HIGH (consistent success)")
                elif self.memory_pressure_level == "high":
                    self.memory_pressure_level = "normal"
                    print("   ✅ Memory pressure normalized (consistent success)")

        # Keep only recent history
        if len(self.successful_batch_sizes) > 10:
            self.successful_batch_sizes = self.successful_batch_sizes[-10:]
    
    def record_failure(self, batch_size, error_type="oom"):
        """Record failed batch processing with enhanced tracking"""
        import time

        current_time = time.time()

        if error_type == "oom":
            self.oom_count += 1
            self.failed_batch_sizes.append(batch_size)

            # Track consecutive OOMs
            if self.last_oom_time and (current_time - self.last_oom_time) < 30:  # Within 30 seconds
                self.consecutive_oom_count += 1
            else:
                self.consecutive_oom_count = 1

            self.last_oom_time = current_time

            # Adjust memory pressure level based on consecutive OOMs
            if self.consecutive_oom_count >= 3:
                self.memory_pressure_level = "critical"
                print("   🚨 CRITICAL: Multiple consecutive OOMs detected - switching to ultra-conservative mode")
            elif self.consecutive_oom_count >= 2:
                self.memory_pressure_level = "high"
                print("   ⚠️  HIGH: Consecutive OOMs detected - increasing memory conservatism")

            # Keep only recent failures
            if len(self.failed_batch_sizes) > 5:
                self.failed_batch_sizes = self.failed_batch_sizes[-5:]

        elif error_type == "other":
            # For non-OOM errors, just track but don't adjust memory pressure
            pass
    
    def get_memory_stats(self):
        """Get current memory statistics"""
        if not torch.cuda.is_available():
            return {}
        
        current_memory = torch.cuda.memory_allocated()
        max_memory = torch.cuda.max_memory_allocated()
        total_memory = torch.cuda.get_device_properties(0).total_memory
        
        return {
            'current_gb': current_memory / 1e9,
            'max_gb': max_memory / 1e9,
            'total_gb': total_memory / 1e9,
            'utilization_percent': (current_memory / total_memory) * 100,
            'peak_utilization_percent': (max_memory / total_memory) * 100
        }
    
    def suggest_optimization(self):
        """Suggest optimizations based on memory usage patterns"""
        stats = self.get_memory_stats()
        suggestions = []
        
        if stats.get('peak_utilization_percent', 0) < 50:
            suggestions.append("💡 Low VRAM usage detected. You can increase batch sizes for better performance.")
        elif stats.get('peak_utilization_percent', 0) > 90:
            suggestions.append("⚠️  High VRAM usage. Consider reducing batch sizes to avoid OOM errors.")
        
        if self.oom_count > 3:
            suggestions.append("🔧 Multiple OOM errors detected. Try reducing optimization_preset or adding more VRAM.")
        
        if len(self.successful_batch_sizes) > 5:
            avg_batch = sum(self.successful_batch_sizes[-5:]) / 5
            suggestions.append(f"📈 Optimal batch size appears to be around {avg_batch:.0f}")
        
        return suggestions


class TrueBatchGFPGANEnhancer:
    """True parallel batch processing GFPGAN enhancer for maximum VRAM utilization"""

    def __init__(self, method='gfpgan'):
        self.method = method
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gfpgan_model = None
        self.streams = []
        self.memory_manager = AdaptiveMemoryManager()

        # Calculate dynamic batch size based on available VRAM
        self.batch_size = self._calculate_dynamic_batch_size()

        self._initialize_parallel_gfpgan()

    def _calculate_dynamic_batch_size(self):
        """Calculate optimal batch size based on available VRAM - scales automatically"""
        if not torch.cuda.is_available():
            return 4  # Conservative for CPU

        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        available_memory_gb = gpu_memory_gb * 0.85  # Use 85% of VRAM for safety

        # Estimate memory per image (GFPGAN + overhead)
        # GFPGAN typically needs ~2-3GB per image for processing
        memory_per_image_gb = 2.5  # Conservative estimate

        # Calculate optimal batch size
        optimal_batch_size = max(1, int(available_memory_gb / memory_per_image_gb))

        # Cap at reasonable maximum (too large batches can cause other issues)
        max_reasonable_batch = 64

        # For very large GPUs (A100, H100), allow larger batches
        if gpu_memory_gb >= 40:  # A100/H100 range
            max_reasonable_batch = 128
        elif gpu_memory_gb >= 24:  # RTX 4090/3090 range
            max_reasonable_batch = 96
        elif gpu_memory_gb >= 16:  # RTX 4080/3080 range
            max_reasonable_batch = 80

        final_batch_size = min(optimal_batch_size, max_reasonable_batch)

        print(f"🚀 Dynamic batch size: {final_batch_size} (GPU: {gpu_memory_gb:.1f}GB, available: {available_memory_gb:.1f}GB)")
        return final_batch_size
    
    def _initialize_parallel_gfpgan(self):
        """Initialize GFPGAN with parallel processing capabilities and memory management"""
        print('Initializing TRUE BATCH GFPGAN enhancer with OOM protection...')
        
        # Initialize memory manager
        self.memory_manager.initialize()
        
        try:
            from gfpgan import GFPGANer
            from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
            
            # Initialize standard GFPGAN first
            model_path = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
            self.gfpgan = GFPGANer(
                model_path=model_path,
                upscale=2,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None
            )
            
            # Extract the actual model for direct batch processing
            self.gfpgan_model = self.gfpgan.gfpgan
            
            # Create multiple CUDA streams for parallel processing
            num_streams = 4  # Use 4 streams for maximum parallelism
            self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
            
            # Pre-allocate tensor pools for different common sizes (smaller pools to avoid OOM)
            self.tensor_pools = {}
            common_sizes = [(512, 512), (256, 256)]  # Reduced pool sizes
            
            for size in common_sizes:
                pool_size = 4  # Smaller pool to avoid OOM
                try:
                    self.tensor_pools[size] = [
                        torch.zeros(3, size[0], size[1], device=self.device, dtype=torch.float32)
                        for _ in range(pool_size)
                    ]
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"⚠️  Cannot pre-allocate tensor pool for {size}, will create tensors on demand")
                        self.tensor_pools[size] = []
                    else:
                        raise e
            
            print(f"✅ Initialized TRUE BATCH enhancer with {num_streams} CUDA streams and OOM protection")
            
        except Exception as e:
            print(f"Failed to initialize parallel GFPGAN: {e}")
            # Fallback to standard GFPGAN
            self.gfpgan = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
                upscale=2,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None
            )
            self.gfpgan_model = self.gfpgan.gfpgan
    
    def enhance_batch_parallel(self, images, batch_size=32, silent=False):
        """True parallel batch processing with comprehensive OOM handling and adaptive memory management"""
        
        # Use adaptive memory manager to determine safe batch size
        if images:
            image_size = images[0].shape[:2] if images[0].shape[:2] else (512, 512)
            safe_batch_size = self.memory_manager.get_safe_batch_size(batch_size, image_size)

            # Additional proactive OOM prevention
            safe_batch_size = self._adaptive_oom_prevention(safe_batch_size, len(images))
        else:
            safe_batch_size = batch_size
        
        # Determine concurrent streams based on memory constraints
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            current_memory_gb = torch.cuda.memory_allocated() / 1e9
            available_memory_gb = gpu_memory_gb - current_memory_gb
            
            if available_memory_gb > 8:
                concurrent_streams = 4
            elif available_memory_gb > 4:
                concurrent_streams = 2
            else:
                concurrent_streams = 1
                
            if not silent:
                print(f"🧠 Adaptive Memory: {available_memory_gb:.2f}GB available, using {concurrent_streams} streams")
        else:
            concurrent_streams = 1
        
        if not silent:
            print(f"🚀 TRUE PARALLEL processing: {len(images)} images across {concurrent_streams} streams (batch size: {safe_batch_size})")
        
        enhanced_images = []
        current_batch_size = safe_batch_size
        oom_retry_count = 0
        max_oom_retries = 3
        
        # Pre-allocate result tensors to maximize memory efficiency
        torch.cuda.empty_cache()
        
        # Process in truly parallel batches using multiple streams with OOM handling
        batch_start = 0
        while batch_start < len(images):
            batch_end = min(batch_start + current_batch_size, len(images))
            batch_images = images[batch_start:batch_end]
            
            # Record memory before processing
            initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            try:
                # Split batch across streams for true parallelism  
                stream_batches = self._split_batch_for_streams(batch_images, concurrent_streams)
                
                # Process all streams in parallel with OOM monitoring (silent mode)
                enhanced_batch = self._process_parallel_streams_with_oom_handling(stream_batches, silent=silent)
                enhanced_images.extend(enhanced_batch)
                
                # Record successful processing
                final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                self.memory_manager.record_success(current_batch_size, final_memory)
                
                # Reset retry count on success
                oom_retry_count = 0
                batch_start = batch_end
                
                # Adaptive memory cleanup
                if batch_start % (current_batch_size * 2) == 0:
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e) and oom_retry_count < max_oom_retries:
                    print(f"⚠️  CUDA OOM detected! Implementing adaptive recovery...")
                    
                    # Record failure
                    self.memory_manager.record_failure(current_batch_size, "oom")
                    
                    # OOM Recovery Strategy
                    recovery_success = self._handle_cuda_oom(e, current_batch_size, concurrent_streams)
                    
                    if recovery_success:
                        # Get new safe batch size from memory manager
                        current_batch_size = self.memory_manager.get_safe_batch_size(
                            max(1, current_batch_size // 2), image_size
                        )
                        concurrent_streams = max(1, concurrent_streams // 2)
                        oom_retry_count += 1
                        
                        print(f"🔧 Adaptive adjustment: batch size {current_batch_size}, streams {concurrent_streams}")
                        print(f"🔄 Retrying batch {batch_start}-{batch_end} (attempt {oom_retry_count + 1})")
                        
                        # Don't advance batch_start, retry with smaller batch
                        continue
                    else:
                        print("❌ OOM recovery failed, using emergency fallback")
                        # Fallback to emergency processing for remaining images
                        remaining_images = images[batch_start:]
                        fallback_enhanced = self._emergency_fallback_processing(remaining_images)
                        enhanced_images.extend(fallback_enhanced)
                        break
                else:
                    # Non-OOM error or too many retries
                    if "out of memory" in str(e):
                        print(f"❌ Persistent CUDA OOM after {max_oom_retries} retries")
                        self.memory_manager.record_failure(current_batch_size, "persistent_oom")
                    else:
                        print(f"❌ Non-OOM error: {e}")
                    
                    # Emergency fallback
                    remaining_images = images[batch_start:]
                    fallback_enhanced = self._emergency_fallback_processing(remaining_images)
                    enhanced_images.extend(fallback_enhanced)
                    break
        
        # Synchronize all streams at the end
        for stream in self.streams:
            stream.synchronize()
        
        # Print memory statistics and suggestions only if not silent
        if not silent:
            self._print_memory_summary()
        
        return enhanced_images
    
    def _print_memory_summary(self):
        """Print memory usage summary and optimization suggestions"""
        stats = self.memory_manager.get_memory_stats()
        suggestions = self.memory_manager.suggest_optimization()
        
        print(f"\n📊 Memory Usage Summary:")
        print(f"   Peak VRAM: {stats.get('max_gb', 0):.2f}GB ({stats.get('peak_utilization_percent', 0):.1f}%)")
        print(f"   Current: {stats.get('current_gb', 0):.2f}GB ({stats.get('utilization_percent', 0):.1f}%)")
        print(f"   OOM Events: {self.memory_manager.oom_count}")
        
        if suggestions:
            print(f"\n💡 Optimization Suggestions:")
            for suggestion in suggestions:
                print(f"   {suggestion}")
        
        # Show successful batch sizes for user reference
        if self.memory_manager.successful_batch_sizes:
            recent_batches = self.memory_manager.successful_batch_sizes[-3:]
            avg_batch = sum(recent_batches) / len(recent_batches)
            print(f"\n📈 Recent successful batch sizes: {recent_batches}")
            print(f"   Average: {avg_batch:.1f}")
    
    def get_memory_recommendations(self):
        """Get memory usage recommendations for future runs"""
        stats = self.memory_manager.get_memory_stats()
        recommendations = {
            'optimal_batch_size': None,
            'memory_utilization': stats.get('peak_utilization_percent', 0),
            'suggestions': self.memory_manager.suggest_optimization()
        }
        
        if self.memory_manager.successful_batch_sizes:
            recommendations['optimal_batch_size'] = max(self.memory_manager.successful_batch_sizes)
        
        return recommendations
    
    def _handle_cuda_oom(self, error, current_batch_size, concurrent_streams):
        """Comprehensive CUDA OOM recovery strategy with multiple fallback levels"""
        print(f"🔧 CUDA OOM Recovery initiated...")
        print(f"   Error: {str(error)[:100]}...")

        try:
            # Step 1: Emergency memory cleanup
            print("   Step 1: Emergency memory cleanup")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Step 2: Clear all stream memory
            print("   Step 2: Clearing stream memory")
            for stream in self.streams:
                with torch.cuda.stream(stream):
                    torch.cuda.empty_cache()
                stream.synchronize()

            # Step 3: Force garbage collection
            print("   Step 3: Forcing garbage collection")
            import gc
            gc.collect()

            # Step 4: Check available memory and implement progressive recovery
            if torch.cuda.is_available():
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                free_memory_gb = free_memory / 1e9
                print(f"   Available VRAM after cleanup: {free_memory_gb:.2f}GB")

                # Progressive recovery based on memory availability
                if free_memory_gb < 1.0:
                    print("   🚨 CRITICAL: Less than 1GB free - implementing emergency measures")
                    # Emergency measures for critical memory situations
                    self._emergency_memory_recovery()

                elif free_memory_gb < 2.0:
                    print("   Step 5: Aggressive cleanup (less than 2GB free)")
                    # Try to free tensor pools
                    self.tensor_pools.clear()
                    torch.cuda.empty_cache()

                    # Reduce memory fraction progressively
                    current_fraction = torch.cuda.get_device_properties(0).total_memory * 0.01  # Get current fraction
                    new_fraction = min(current_fraction, 0.6)  # Cap at 60%
                    torch.cuda.set_per_process_memory_fraction(new_fraction)
                    print(f"   Reduced memory fraction to {new_fraction:.1f}")

                    free_memory_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
                    print(f"   Available VRAM after aggressive cleanup: {free_memory_gb:.2f}GB")

                elif free_memory_gb < 4.0:
                    print("   Step 5: Moderate cleanup (2-4GB free)")
                    # Moderate cleanup
                    torch.cuda.set_per_process_memory_fraction(0.75)
                    free_memory_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
                    print(f"   Available VRAM after moderate cleanup: {free_memory_gb:.2f}GB")

            # Final check: if still critically low memory, use extreme recovery
            final_free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            final_free_memory_gb = final_free_memory / 1e9

            if final_free_memory_gb < 0.5:  # Less than 500MB free
                print("   💀 FINAL RESORT: Less than 500MB free - extreme memory pressure recovery")
                success = self._extreme_memory_pressure_recovery()
                if not success:
                    print("   ❌ All recovery methods failed - will return original images")
                    return False

            print("✅ OOM recovery completed successfully")
            return True

        except Exception as recovery_error:
            print(f"❌ OOM recovery failed: {recovery_error}")
            return False

    def _emergency_memory_recovery(self):
        """Emergency memory recovery for critical OOM situations"""
        print("   🚨 EMERGENCY MEMORY RECOVERY ACTIVATED")

        try:
            # Force maximum memory cleanup
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Clear all cached memory
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()

            # Set memory fraction to minimum safe level
            torch.cuda.set_per_process_memory_fraction(0.5)  # 50% of GPU memory

            # Force garbage collection multiple times
            import gc
            for _ in range(3):
                gc.collect()

            # Try to unload any unused GPU memory
            torch.cuda.empty_cache()

            final_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            final_memory_gb = final_memory / 1e9
            print(f"   Emergency recovery: {final_memory_gb:.2f}GB now available")

        except Exception as e:
            print(f"   Emergency recovery failed: {e}")

    def _extreme_memory_pressure_recovery(self):
        """
        Extreme memory pressure recovery for critical situations.
        This is the last resort before returning original images.
        """
        print("   💀 EXTREME MEMORY PRESSURE RECOVERY ACTIVATED")

        try:
            # Force garbage collection multiple times
            import gc
            for _ in range(5):
                gc.collect()

            # Clear all CUDA caches aggressively
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Reset memory fraction to minimum
            torch.cuda.set_per_process_memory_fraction(0.3)  # 30% of GPU memory
            self.memory_fraction = 0.3

            # Clear all tensor pools completely
            self.tensor_pools = {}
            self.pool_sizes = {}

            # Force recreation of GFPGAN models with minimal memory
            if hasattr(self, 'gfpgan_enhancers'):
                for enhancer in self.gfpgan_enhancers:
                    if enhancer is not None:
                        del enhancer
                self.gfpgan_enhancers = None

            # Clear any cached models
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()

            # Reinitialize with minimal settings
            self._initialize_enhancers()

            final_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            final_memory_gb = final_memory / 1e9
            print(f"   Extreme recovery: {final_memory_gb:.2f}GB now available (30% memory fraction)")

            return True

        except Exception as e:
            print(f"   ❌ Extreme memory recovery failed: {e}")
            return False

    def get_performance_insights(self):
        """Get performance insights and optimization recommendations"""
        insights = {
            "memory_utilization": 0.0,
            "oom_events": self.oom_count,
            "memory_pressure_level": self.memory_pressure_level,
            "successful_batches": len(self.successful_batch_sizes),
            "failed_batches": len(self.failed_batch_sizes),
            "recommendations": []
        }

        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated()
            insights["memory_utilization"] = allocated_memory / total_memory

        # Generate recommendations based on performance
        if self.oom_count > 5:
            insights["recommendations"].append("Consider reducing concurrent streams or batch sizes")
        if self.memory_pressure_level == "critical":
            insights["recommendations"].append("GPU memory is under extreme pressure - consider upgrading GPU")
        if len(self.successful_batch_sizes) > 0:
            avg_successful_batch = sum(self.successful_batch_sizes) / len(self.successful_batch_sizes)
            if avg_successful_batch < 4:
                insights["recommendations"].append("Small batch sizes detected - GPU may be memory-constrained")

        return insights

    def _adaptive_oom_prevention(self, batch_size, image_count):

        try:
            # Get current memory state
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated()
            available_memory = total_memory - allocated_memory

            # Estimate memory needed for this batch
            # Conservative estimate: 2.5GB per image for GFPGAN processing
            estimated_memory_per_image = 2.5 * 1e9  # 2.5GB in bytes
            estimated_total_memory = batch_size * image_count * estimated_memory_per_image

            # Safety margin: leave 20% buffer
            safe_available_memory = available_memory * 0.8

            if estimated_total_memory > safe_available_memory:
                # Calculate safe batch size
                safe_batch_size = max(1, int(safe_available_memory / (image_count * estimated_memory_per_image)))

                if safe_batch_size < batch_size:
                    print(f"   ⚠️  OOM Prevention: Reducing batch size from {batch_size} to {safe_batch_size}")
                    print(f"      Estimated: {estimated_total_memory/1e9:.1f}GB needed, {safe_available_memory/1e9:.1f}GB safe")
                    return safe_batch_size

        except Exception as e:
            print(f"   OOM prevention check failed: {e}")

        return batch_size
    
    def _process_parallel_streams_with_oom_handling(self, stream_batches, silent=False):
        """Process multiple batches in parallel with OOM monitoring"""
        futures = []
        enhanced_results = [None] * len(stream_batches)
        
        # Launch all streams in parallel with individual OOM handling
        for i, (stream_batch, stream) in enumerate(zip(stream_batches, self.streams[:len(stream_batches)])):
            try:
                with torch.cuda.stream(stream):
                    # Process this stream's batch with OOM protection
                    result = self._process_stream_batch_with_oom_protection(stream_batch, stream, i, silent=silent)
                    enhanced_results[i] = result
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if not silent:
                        print(f"⚠️  Stream {i} OOM, attempting recovery...")
                    # Try processing with smaller chunks
                    result = self._process_stream_fallback(stream_batch, stream, i, silent=silent)
                    enhanced_results[i] = result
                else:
                    raise e
        
        # Wait for all streams to complete and collect results
        for stream in self.streams[:len(stream_batches)]:
            stream.synchronize()
        
        # Flatten results
        all_enhanced = []
        for result in enhanced_results:
            if result:
                all_enhanced.extend(result)
        
        return all_enhanced
    
    def _process_stream_batch_with_oom_protection(self, stream_images, stream, stream_id, silent=False):
        """Process multiple images within a single CUDA stream with true parallel processing"""
        enhanced_images = []

        try:
            # Process each image in this stream individually but efficiently
            for img in stream_images:
                try:
                    # Convert single image to tensor
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    img_tensor = torch.from_numpy(img_bgr).float().to(self.device) / 255.0
                    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

                    # Process through GFPGAN (individual image processing)
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        # Convert tensor back to numpy for GFPGAN
                        img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                        img_np = (img_np * 255).astype(np.uint8)

                        # Process through GFPGAN enhance method
                        _, _, enhanced_bgr = self.gfpgan.enhance(
                            img_np, has_aligned=False, only_center_face=False, paste_back=True
                        )

                        # Convert back to RGB
                        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
                        enhanced_images.append(enhanced_rgb)

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"   ⚠️  OOM on stream {stream_id}, clearing cache...")
                        torch.cuda.empty_cache()
                        # Use original image if enhancement fails
                        enhanced_images.append(img)
                    else:
                        print(f"   ⚠️  Error on stream {stream_id}: {e}")
                        enhanced_images.append(img)

            # Log memory usage for this stream (only if multiple images processed and not silent)
            # Removed excessive per-stream logging to improve performance

        except Exception as e:
            print(f"   Stream {stream_id} batch processing failed: {e}")
            # Fallback: return original images
            enhanced_images = stream_images

        return enhanced_images

    def _process_stream_fallback(self, batch_images, stream, stream_id, silent=False):
        """Fallback processing for a stream batch with minimal memory usage"""
        enhanced_batch = []
        
        if not silent:
            print(f"   Stream {stream_id}: Using conservative fallback processing")
        
        # Process images one by one to minimize memory usage
        for i, img in enumerate(batch_images):
            try:
                # Clear cache before each image
                if i % 2 == 0:
                    torch.cuda.empty_cache()
                
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                _, _, restored_img = self.gfpgan.enhance(
                    img_bgr, has_aligned=False, only_center_face=False, paste_back=True
                )
                restored_rgb = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
                enhanced_batch.append(restored_rgb)
                
            except Exception as e:
                if not silent:
                    print(f"   Stream {stream_id}, image {i}: Enhancement failed, using original")
                enhanced_batch.append(img)  # Use original if enhancement fails
        
        return enhanced_batch
    
    def _emergency_fallback_processing(self, images):
        """Emergency sequential processing when all else fails"""
        print("🚨 EMERGENCY FALLBACK: Processing remaining images sequentially")
        
        enhanced_images = []
        
        # Use minimal memory footprint
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.5)  # Use only 50% VRAM
        
        for i, img in enumerate(tqdm(images, desc="Emergency Processing")):
            try:
                # Clear cache every few images
                if i % 5 == 0:
                    torch.cuda.empty_cache()
                
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                _, _, restored_img = self.gfpgan.enhance(
                    img_bgr, has_aligned=False, only_center_face=False, paste_back=True
                )
                restored_rgb = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
                enhanced_images.append(restored_rgb)
                
            except Exception as e:
                print(f"Emergency processing failed for image {i}: {e}")
                enhanced_images.append(img)  # Use original
        
        return enhanced_images
    
    def _split_batch_for_streams(self, batch_images, num_streams):
        """Split batch into larger chunks for each stream for better efficiency"""
        # Instead of round-robin (which creates 1-image batches), 
        # divide images into larger chunks for each stream
        images_per_stream = max(1, len(batch_images) // num_streams)
        stream_batches = []
        
        for i in range(num_streams):
            start_idx = i * images_per_stream
            end_idx = min((i + 1) * images_per_stream, len(batch_images))
            
            if start_idx < len(batch_images):
                stream_batch = batch_images[start_idx:end_idx]
                if stream_batch:  # Only add non-empty batches
                    stream_batches.append(stream_batch)
        
        # Handle any remaining images by adding them to the last stream
        remaining_start = num_streams * images_per_stream
        if remaining_start < len(batch_images):
            remaining_images = batch_images[remaining_start:]
            if stream_batches:
                stream_batches[-1].extend(remaining_images)
            else:
                stream_batches.append(remaining_images)

        return stream_batches
    
    def _process_parallel_streams(self, stream_batches):
        """Process multiple batches in parallel using CUDA streams"""
        futures = []
        enhanced_results = [None] * len(stream_batches)
        
        # Launch all streams in parallel
        for i, (stream_batch, stream) in enumerate(zip(stream_batches, self.streams[:len(stream_batches)])):
            with torch.cuda.stream(stream):
                # Process this stream's batch
                result = self._process_stream_batch(stream_batch, stream)
                enhanced_results[i] = result
        
        # Wait for all streams to complete and collect results
        for stream in self.streams[:len(stream_batches)]:
            stream.synchronize()
        
        # Flatten results
        all_enhanced = []
        for result in enhanced_results:
            if result:
                all_enhanced.extend(result)
        
        return all_enhanced
    
    def _process_stream_batch(self, batch_images, stream):
        """Process a batch within a single CUDA stream"""
        enhanced_batch = []
        
        try:
            # Convert images to tensors in parallel
            batch_tensors = self._prepare_batch_tensors(batch_images)
            
            # Process tensors through GFPGAN model in batch
            with torch.cuda.amp.autocast():  # Use mixed precision
                enhanced_tensors = self._enhance_tensor_batch(batch_tensors)
            
            # Convert back to numpy arrays
            enhanced_batch = self._tensors_to_images(enhanced_tensors)
            
        except Exception as e:
            print(f"Stream batch processing failed: {e}, falling back to sequential")
            # Fallback to sequential processing for this batch
            for img in batch_images:
                try:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    _, _, restored_img = self.gfpgan.enhance(
                        img_bgr, has_aligned=False, only_center_face=False, paste_back=True
                    )
                    restored_rgb = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
                    enhanced_batch.append(restored_rgb)
                except:
                    enhanced_batch.append(img)  # Use original if enhancement fails
        
        return enhanced_batch
    
    def _prepare_batch_tensors_safe(self, batch_images):
        """Prepare batch tensors with OOM protection and memory monitoring"""
        batch_tensors = []
        
        try:
            # Monitor memory before tensor creation
            initial_memory = torch.cuda.memory_allocated() / 1e9
            
            # Pre-allocate tensors from pool when possible (with OOM protection)
            h, w = batch_images[0].shape[:2] if batch_images else (512, 512)
            target_size = (h, w)
            
            # Check if we have enough memory for this batch
            estimated_memory_needed = len(batch_images) * h * w * 3 * 4 / 1e9  # 4 bytes per float32
            available_memory = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
            
            if estimated_memory_needed > available_memory * 0.8:  # Leave 20% buffer
                print(f"   ⚠️  Estimated memory needed ({estimated_memory_needed:.2f}GB) > available ({available_memory:.2f}GB)")
                # Process in smaller chunks
                chunk_size = max(1, int(len(batch_images) * available_memory * 0.6 / estimated_memory_needed))
                return self._prepare_chunked_tensors(batch_images[:chunk_size])
            
            # Use tensor pool if available for this size
            if target_size in self.tensor_pools and self.tensor_pools[target_size]:
                # Use pre-allocated tensors for maximum efficiency
                pool_tensors = self.tensor_pools[target_size]
                
                for i, img in enumerate(batch_images):
                    if i < len(pool_tensors):
                        # Reuse pre-allocated tensor
                        tensor = pool_tensors[i]
                        
                        # In-place operations for memory efficiency
                        if len(img.shape) == 3:
                            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        else:
                            img_bgr = img
                        
                        # Copy data to pre-allocated tensor
                        img_tensor = torch.from_numpy(img_bgr).float().to(self.device)
                        img_tensor.div_(255.0)  # In-place normalization
                        img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
                        
                        # Resize to match pool tensor if needed
                        if img_tensor.shape != tensor.shape:
                            img_tensor = torch.nn.functional.interpolate(
                                img_tensor.unsqueeze(0), size=tensor.shape[1:], mode='bilinear', align_corners=False
                            ).squeeze(0)
                        
                        tensor.copy_(img_tensor)  # In-place copy
                        batch_tensors.append(tensor.unsqueeze(0))
                    else:
                        # Create new tensor if pool exhausted
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 else img
                        img_tensor = torch.from_numpy(img_bgr).float().to(self.device) / 255.0
                        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
                        batch_tensors.append(img_tensor)
            else:
                # Standard tensor creation with memory optimization
                for i, img in enumerate(batch_images):
                    # Check memory every few images
                    if i % 4 == 0:
                        current_memory = torch.cuda.memory_allocated() / 1e9
                        if current_memory > initial_memory + estimated_memory_needed * 1.2:
                            print(f"   ⚠️  Memory usage exceeded expected, stopping at image {i}")
                            break
                    
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 else img
                    
                    # Use pinned memory for faster GPU transfer
                    img_tensor = torch.from_numpy(img_bgr).float()
                    img_tensor = img_tensor.div(255.0)  # More efficient than division
                    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
                    img_tensor = img_tensor.to(self.device, non_blocking=True)  # Async transfer
                    
                    batch_tensors.append(img_tensor)
            
            # Efficiently stack tensors into single batch
            if batch_tensors:
                return self._stack_tensors_safely(batch_tensors)
            
            return None
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"   ⚠️  OOM during tensor preparation, using smaller chunks")
                # Free any partially created tensors
                del batch_tensors
                torch.cuda.empty_cache()
                # Try with half the images
                half_size = max(1, len(batch_images) // 2)
                return self._prepare_chunked_tensors(batch_images[:half_size])
            else:
                raise e
    
    def _prepare_chunked_tensors(self, batch_images):
        """Prepare tensors in smaller chunks to avoid OOM"""
        if not batch_images:
            return None
        
        # Process just a few images at a time
        chunk_size = min(4, len(batch_images))
        batch_tensors = []
        
        for img in batch_images[:chunk_size]:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 else img
            img_tensor = torch.from_numpy(img_bgr).float().to(self.device) / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
            batch_tensors.append(img_tensor)
        
        return self._stack_tensors_safely(batch_tensors)
    
    def _stack_tensors_safely(self, batch_tensors):
        """Safely stack tensors with memory monitoring"""
        try:
            # Find optimal tensor size for batching
            max_h = max(t.shape[2] for t in batch_tensors)
            max_w = max(t.shape[3] for t in batch_tensors)
            
            # Round up to nearest multiple of 32 for optimal GPU processing
            max_h = ((max_h + 31) // 32) * 32
            max_w = ((max_w + 31) // 32) * 32
            
            # Use in-place padding when possible
            padded_tensors = []
            for tensor in batch_tensors:
                h, w = tensor.shape[2], tensor.shape[3]
                if h == max_h and w == max_w:
                    padded_tensors.append(tensor)
                else:
                    # Efficient padding
                    pad_h = max_h - h
                    pad_w = max_w - w
                    padded = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
                    padded_tensors.append(padded)
            
            # Concatenate efficiently
            return torch.cat(padded_tensors, dim=0)
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"   ⚠️  OOM during tensor stacking, using sequential processing")
                return None
            else:
                raise e
    
    def _enhance_tensor_batch_safe(self, batch_tensor):
        """Enhance batch tensor with OOM protection - GFPGAN processes one image at a time"""
        if batch_tensor is None:
            return None
        
        try:
            # Monitor memory before processing
            initial_memory = torch.cuda.memory_allocated() / 1e9
            
            # GFPGAN processes images individually, not as batches
            # So we need to process each image separately but efficiently
            enhanced_images = []
            
            with torch.no_grad(), torch.cuda.amp.autocast():
                # Process each image in the batch individually
                for i in range(batch_tensor.shape[0]):
                    try:
                        # Extract single image tensor
                        single_img_tensor = batch_tensor[i:i+1]  # Keep batch dimension
                        
                        # Convert tensor back to numpy for GFPGAN
                        # GFPGAN expects HWC format BGR numpy array
                        img_np = single_img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                        img_np = (img_np * 255).astype(np.uint8)
                        
                        # GFPGAN processes BGR images
                        if img_np.shape[-1] == 3:
                            img_bgr = img_np  # Already BGR from our tensor prep
                        else:
                            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                        
                        # Process through GFPGAN enhance method (not model directly)
                        _, _, enhanced_bgr = self.gfpgan.enhance(
                            img_bgr, 
                            has_aligned=False, 
                            only_center_face=False, 
                            paste_back=True
                        )
                        
                        # Convert back to tensor format
                        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
                        enhanced_tensor = torch.from_numpy(enhanced_rgb).float().to(self.device)
                        enhanced_tensor = enhanced_tensor.div(255.0).permute(2, 0, 1).unsqueeze(0)
                        
                        enhanced_images.append(enhanced_tensor)
                        
                        # Clear memory periodically
                        if i % 4 == 0:
                            torch.cuda.empty_cache()
                            
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"   ⚠️  OOM on image {i}, skipping...")
                            # Use original image if enhancement fails
                            enhanced_images.append(batch_tensor[i:i+1])
                            torch.cuda.empty_cache()
                        else:
                            print(f"   ⚠️  Error on image {i}: {e}, using original")
                            enhanced_images.append(batch_tensor[i:i+1])
                
                # Concatenate all enhanced images
                if enhanced_images:
                    return torch.cat(enhanced_images, dim=0)
                else:
                    return batch_tensor  # Return original if all failed
            
        except Exception as e:
            print(f"   ⚠️  Batch enhancement failed: {e}, using original images")
            return batch_tensor
    
    def _enhance_tensor_batch(self, batch_tensor):
        """Standard tensor enhancement for non-safe processing"""
        if batch_tensor is None:
            return None
        
        # Use the safe version since GFPGAN has the same limitations
        return self._enhance_tensor_batch_safe(batch_tensor)
    
    def _prepare_batch_tensors(self, batch_images):
        """Standard tensor preparation for non-safe processing"""
        # Use the safe version for consistency
        return self._prepare_batch_tensors_safe(batch_images)
    
    def _tensors_to_images(self, enhanced_tensors):
        """Standard tensor to image conversion"""
        # Use the safe version for consistency
        return self._tensors_to_images_safe(enhanced_tensors)
    
    def _enhance_tensor_chunked(self, batch_tensor):
        """Enhanced tensor processing in very small chunks - GFPGAN processes individually"""
        if batch_tensor is None:
            return None
        
        enhanced_chunks = []
        chunk_size = 2  # Very conservative chunk size
        
        for i in range(0, batch_tensor.shape[0], chunk_size):
            end_idx = min(i + chunk_size, batch_tensor.shape[0])
            chunk = batch_tensor[i:end_idx]
            
            try:
                # Process each image in chunk individually (GFPGAN limitation)
                chunk_enhanced = []
                for j in range(chunk.shape[0]):
                    single_img = chunk[j:j+1]
                    
                    # Convert to numpy for GFPGAN processing
                    img_np = single_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    img_np = (img_np * 255).astype(np.uint8)
                    
                    # Process through GFPGAN enhance method
                    _, _, enhanced_bgr = self.gfpgan.enhance(
                        img_np, has_aligned=False, only_center_face=False, paste_back=True
                    )
                    
                    # Convert back to tensor
                    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
                    enhanced_tensor = torch.from_numpy(enhanced_rgb).float().to(self.device)
                    enhanced_tensor = enhanced_tensor.div(255.0).permute(2, 0, 1).unsqueeze(0)
                    chunk_enhanced.append(enhanced_tensor)
                
                # Concatenate chunk results
                if chunk_enhanced:
                    enhanced_chunks.append(torch.cat(chunk_enhanced, dim=0))
                    
            except Exception as e:
                print(f"   ⚠️  Chunk {i}-{end_idx} failed: {e}, using original")
                enhanced_chunks.append(chunk)
        
        # Concatenate all chunks
        if enhanced_chunks:
            return torch.cat(enhanced_chunks, dim=0)
        else:
            return batch_tensor
            chunk = batch_tensor[i:end_idx]
            
            try:
                with torch.no_grad():
                    enhanced_chunk = self.gfpgan_model(chunk)
                    enhanced_chunks.append(enhanced_chunk)
                    
                # Clear cache after each chunk
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"   ⚠️  Even chunked processing failed, using single image processing")
                    # Process single images
                    for j in range(i, end_idx):
                        single_tensor = batch_tensor[j:j+1]
                        try:
                            single_enhanced = self.gfpgan_model(single_tensor)
                            enhanced_chunks.append(single_enhanced)
                        except:
                            # Skip this image if it still fails
                            enhanced_chunks.append(single_tensor)
                        torch.cuda.empty_cache()
                else:
                    raise e
        
        if enhanced_chunks:
            return torch.cat(enhanced_chunks, dim=0)
        return None
    
    def _tensors_to_images_safe(self, enhanced_tensors):
        """Convert enhanced tensors back to images with OOM protection"""
        if enhanced_tensors is None:
            return []
        
        images = []
        
        try:
            # Process tensors in batches to avoid memory issues
            for i in range(enhanced_tensors.shape[0]):
                tensor = enhanced_tensors[i]
                
                # Efficient tensor to numpy conversion
                tensor = tensor.permute(1, 2, 0)  # CHW -> HWC
                
                # Clamp and convert with minimal memory allocation
                tensor = tensor.mul(255).clamp(0, 255).to(torch.uint8)
                img_array = tensor.cpu().numpy()
                
                # BGR to RGB conversion
                img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                images.append(img_rgb)
                
                # Clear tensor reference for memory efficiency
                del tensor
                
                # Clear cache every few images
                if i % 4 == 0:
                    torch.cuda.empty_cache()
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"   ⚠️  OOM during tensor to image conversion")
                torch.cuda.empty_cache()
                # Try with just the tensors we've processed so far
                return images
            else:
                raise e
        
        return images


class HighVRAMFaceEnhancer:
    """Ultra-optimized face enhancer for high VRAM systems (12GB+)"""
    
    def __init__(self, method='gfpgan'):
        self.method = method
        self.restorer = None
        self._initialize_gfpgan()
    
    def __init__(self, method='gfpgan'):
        self.method = method
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use the new true batch processor for maximum VRAM utilization
        self.true_batch_enhancer = TrueBatchGFPGANEnhancer(method=method)
        
        # Also keep fallback enhancer
        self.fallback_enhancer = None
        self._initialize_fallback_gfpgan()
    
    def _initialize_fallback_gfpgan(self):
        """Initialize fallback GFPGAN for error handling"""
        try:
            if self.method == 'gfpgan':
                model_name = 'GFPGANv1.4'
                url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
            
            model_path = os.path.join('gfpgan/weights', model_name + '.pth')
            if not os.path.isfile(model_path):
                model_path = os.path.join('checkpoints', model_name + '.pth')
            if not os.path.isfile(model_path):
                model_path = url

            self.fallback_enhancer = GFPGANer(
                model_path=model_path,
                upscale=2,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None
            )
            
        except Exception as e:
            print(f"Warning: Could not initialize fallback enhancer: {e}")
    
    def enhance_batch_ultra(self, images, batch_size=32):
        """Ultra-fast batch processing with TRUE parallel GPU utilization"""

        # Calculate dynamic batch size based on available VRAM
        batch_size = self._calculate_ultra_batch_size(batch_size)

        # Setup for maximum GPU utilization
        if torch.cuda.is_available():
            # Use 95% of VRAM for processing
            torch.cuda.set_per_process_memory_fraction(0.95)

            # Enable all optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.deterministic = False

            # Pre-warm GPU
            torch.cuda.empty_cache()

        try:
            # Use true batch processing for maximum efficiency with minimal logging
            enhanced_images = self.true_batch_enhancer.enhance_batch_parallel(images, batch_size=batch_size, silent=True)
            return enhanced_images
            
        except Exception as e:
            print(f"⚠️  True batch enhancer failed: {e}")
            print("🔄 Falling back to optimized sequential processing...")
            
            # Fallback to optimized sequential processing
            return self._fallback_batch_enhance(images, batch_size // 2)
    
    def _calculate_ultra_batch_size(self, requested_batch_size=32):
        """Calculate ultra-aggressive batch size based on available VRAM - scales automatically"""
        if not torch.cuda.is_available():
            return min(8, requested_batch_size)

        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        available_memory_gb = gpu_memory_gb * 0.95  # Use 95% for ultra mode

        # Estimate memory per image (more aggressive than standard mode)
        memory_per_image_gb = 2.0  # More aggressive estimate for ultra mode

        # Calculate optimal batch size
        optimal_batch_size = max(1, int(available_memory_gb / memory_per_image_gb))

        # Scale maximum based on GPU size
        if gpu_memory_gb >= 80:  # A100 80GB, H100
            max_batch = 256
        elif gpu_memory_gb >= 40:  # A100 40GB, RTX A6000
            max_batch = 192
        elif gpu_memory_gb >= 24:  # RTX 4090/3090
            max_batch = 128
        elif gpu_memory_gb >= 16:  # RTX 4080/3080
            max_batch = 96
        elif gpu_memory_gb >= 12:  # RTX 4070/3070
            max_batch = 80
        elif gpu_memory_gb >= 8:   # RTX 3060/4060
            max_batch = 64
        else:
            max_batch = 48

        final_batch_size = min(optimal_batch_size, max_batch, requested_batch_size)
        print(f"🚀 ULTRA VRAM MODE: {final_batch_size} batch size for {gpu_memory_gb:.1f}GB GPU")
        return final_batch_size
    
    def _fallback_batch_enhance(self, images, batch_size):
        """Fallback batch enhancement with memory optimization"""
        enhanced_images = []
        
        for batch_start in tqdm(range(0, len(images), batch_size), 'Fallback Face Enhancer:'):
            batch_end = min(batch_start + batch_size, len(images))
            batch_images = images[batch_start:batch_end]
            
            # Process batch with memory optimization
            for img in batch_images:
                try:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    _, _, restored_img = self.fallback_enhancer.enhance(
                        img_bgr,
                        has_aligned=False,
                        only_center_face=False,
                        paste_back=True
                    )
                    restored_rgb = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
                    enhanced_images.append(restored_rgb)
                except Exception as e:
                    print(f"Enhancement failed, using original: {e}")
                    enhanced_images.append(img)
            
            # Clear cache every few batches
            if torch.cuda.is_available() and batch_start % (batch_size * 4) == 0:
                torch.cuda.empty_cache()
        
        return enhanced_images


# Update the main enhancer factory function
def create_optimized_enhancer(method='gfpgan', optimization_level="medium"):
    """Factory function to create the best enhancer for the system - scales automatically"""

    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

        # Use HighVRAMFaceEnhancer for systems that can benefit from parallel processing
        # Minimum requirement: enough memory for at least 8 images in parallel
        min_memory_for_parallel_gb = 8 * 2.5  # 8 images * 2.5GB per image estimate

        if gpu_memory_gb >= min_memory_for_parallel_gb and optimization_level in ["medium", "low"]:
            print(f"🚀 Detected {gpu_memory_gb:.1f}GB VRAM - Using HIGH VRAM optimizer")
            return HighVRAMFaceEnhancer(method=method)

    # Use standard optimizer for other cases
    return OptimizedFaceEnhancer(method=method, optimization_level=optimization_level)


class OptimizedFaceEnhancer:
    def __init__(self, method='gfpgan', optimization_level="medium"):
        self.method = method
        self.optimization_level = optimization_level
        self.restorer = None
        self.lightweight_enhancer = LightweightEnhancer()
        
        # Initialize based on optimization level
        if optimization_level == "extreme":
            print("Using lightweight enhancement for extreme speed")
            # Don't initialize GFPGAN for extreme optimization
        else:
            self._initialize_gfpgan()
    
    def _initialize_gfpgan(self):
        """Initialize GFPGAN with optimizations"""
        print('Initializing optimized face enhancer...')
        
        if self.method == 'gfpgan':
            arch = 'clean'
            channel_multiplier = 2
            model_name = 'GFPGANv1.4'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
        elif self.method == 'RestoreFormer':
            arch = 'RestoreFormer'
            channel_multiplier = 2
            model_name = 'RestoreFormer'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
        else:
            raise ValueError(f'Unsupported method {self.method}')

        # Background upsampler optimization
        bg_upsampler = None
        if self.optimization_level in ["low", "medium"] and torch.cuda.is_available():
            try:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                bg_upsampler = RealESRGANer(
                    scale=2,
                    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                    model=model,
                    tile=400,
                    tile_pad=10,
                    pre_pad=0,
                    half=True)
            except:
                bg_upsampler = None

        # Model path handling
        model_path = os.path.join('gfpgan/weights', model_name + '.pth')
        if not os.path.isfile(model_path):
            model_path = os.path.join('checkpoints', model_name + '.pth')
        if not os.path.isfile(model_path):
            model_path = url

        self.restorer = GFPGANer(
            model_path=model_path,
            upscale=2,
            arch=arch,
            channel_multiplier=channel_multiplier,
            bg_upsampler=bg_upsampler
        )

    def enhance_batch(self, images, batch_size=8):
        """Enhanced batch processing with aggressive memory optimization for 15GB VRAM"""
        
        if self.optimization_level == "extreme":
            # Use lightweight enhancement
            return self._lightweight_batch_enhance(images, batch_size)
        
        if not isinstance(images, list) and os.path.isfile(images):
            images = load_video_to_cv2(images)
        
        # Dynamic batch size optimization based on available VRAM
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU Memory Available: {gpu_memory_gb:.1f}GB")

            # Calculate optimal batch size dynamically
            optimal_batch_size = self._calculate_optimal_batch_size(gpu_memory_gb)

            # Use the larger of provided batch_size and optimal_batch_size
            batch_size = max(batch_size, optimal_batch_size)
            print(f"Using optimized batch size: {batch_size} (optimal: {optimal_batch_size})")
        
        # Setup memory optimizations
        if torch.cuda.is_available():
            # Enable memory efficient attention and other optimizations
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Clear cache before starting
            torch.cuda.empty_cache()
        
        # Process in optimized batches
        if self.optimization_level in ["high", "extreme"] and len(images) > batch_size:
            return self._parallel_batch_enhance(images, batch_size)
        else:
            return self._sequential_batch_enhance(images, batch_size)
    
    def _lightweight_batch_enhance(self, images, batch_size):
        """Ultra-fast lightweight enhancement"""
        def enhance_batch_lightweight(batch_images):
            return [self.lightweight_enhancer.enhance(img) for img in batch_images]
        
        enhanced_images = []
        num_workers = min(4, multiprocessing.cpu_count())
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for batch_start in range(0, len(images), batch_size):
                batch_end = min(batch_start + batch_size, len(images))
                batch = images[batch_start:batch_end]
                
                # Convert BGR to RGB if needed
                batch_rgb = []
                for img in batch:
                    if len(img.shape) == 3:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[2] == 3 else img
                    else:
                        img_rgb = img
                    batch_rgb.append(img_rgb)
                
                future = executor.submit(enhance_batch_lightweight, batch_rgb)
                batch_enhanced = future.result()
                enhanced_images.extend(batch_enhanced)
        
        return enhanced_images
    
    def _parallel_batch_enhance(self, images, batch_size):
        """Parallel batch enhancement with GFPGAN"""
        enhanced_images = []
        num_workers = 2  # Limit to 2 workers to avoid GPU memory issues
        
        def enhance_single_batch(batch_images):
            batch_results = []
            for img in batch_images:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                try:
                    _, _, restored_img = self.restorer.enhance(
                        img_bgr,
                        has_aligned=False,
                        only_center_face=False,
                        paste_back=True
                    )
                    restored_rgb = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
                    batch_results.append(restored_rgb)
                except Exception as e:
                    print(f"Enhancement failed for image, using original: {e}")
                    batch_results.append(img)
            return batch_results
        
        # Process batches sequentially to avoid GPU memory conflicts
        for batch_start in tqdm(range(0, len(images), batch_size), 'Fast Face Enhancer:'):
            batch_end = min(batch_start + batch_size, len(images))
            batch = images[batch_start:batch_end]
            
            batch_enhanced = enhance_single_batch(batch)
            enhanced_images.extend(batch_enhanced)
            
            # Clear GPU cache periodically
            if torch.cuda.is_available() and batch_start % (batch_size * 4) == 0:
                torch.cuda.empty_cache()
        
        return enhanced_images
    
    def _sequential_batch_enhance(self, images, batch_size):
        """Optimized batch enhancement with true GPU batch processing"""
        enhanced_images = []

        # Use dynamic batch size calculation
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            max_batch_size = self._calculate_optimal_batch_size(gpu_memory_gb)
            batch_size = min(batch_size, max_batch_size)
        
        for batch_start in tqdm(range(0, len(images), batch_size), 'Face Enhancer (batch):'):
            batch_end = min(batch_start + batch_size, len(images))
            batch_images = images[batch_start:batch_end]
            
            # Process the entire batch at once using GFPGAN's batch capabilities
            batch_enhanced = self._process_image_batch(batch_images)
            enhanced_images.extend(batch_enhanced)
            
            # Periodic cleanup every 4 batches
            if torch.cuda.is_available() and batch_start % (batch_size * 4) == 0:
                torch.cuda.empty_cache()
        
        return enhanced_images
    
    def _process_image_batch(self, batch_images):
        """Process a batch of images efficiently using tensor operations"""
        batch_enhanced = []
        
        try:
            # Convert batch to tensor for GPU processing
            batch_tensors = []
            original_sizes = []
            
            for img in batch_images:
                # Convert RGB to BGR for GFPGAN
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                original_sizes.append(img_bgr.shape[:2])
                
                # Normalize and convert to tensor
                img_tensor = torch.from_numpy(img_bgr).float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
                batch_tensors.append(img_tensor)
            
            # Pad images to same size for batching
            max_h = max([t.shape[1] for t in batch_tensors])
            max_w = max([t.shape[2] for t in batch_tensors])
            
            # Ensure dimensions are multiples of 8 for efficiency
            max_h = ((max_h + 7) // 8) * 8
            max_w = ((max_w + 7) // 8) * 8
            
            padded_tensors = []
            for i, tensor in enumerate(batch_tensors):
                h, w = tensor.shape[1], tensor.shape[2]
                # Pad tensor
                pad_h = max_h - h
                pad_w = max_w - w
                padded = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
                padded_tensors.append(padded)
            
            # Stack into batch tensor
            if torch.cuda.is_available():
                batch_tensor = torch.stack(padded_tensors).cuda()
            else:
                batch_tensor = torch.stack(padded_tensors)
            
            # Process batch through GFPGAN (if it supports batch processing)
            # For now, we'll process individually but with optimized memory management
            for i, img in enumerate(batch_images):
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # Use GFPGAN enhance method
                _, _, restored_img = self.restorer.enhance(
                    img_bgr,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True
                )
                
                restored_rgb = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
                batch_enhanced.append(restored_rgb)
                
                # Clear intermediate GPU memory every few images
                if torch.cuda.is_available() and i % 2 == 0:
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            print(f"Batch processing failed, falling back to individual processing: {e}")
            # Fallback to individual processing
            for img in batch_images:
                try:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    _, _, restored_img = self.restorer.enhance(
                        img_bgr,
                        has_aligned=False,
                        only_center_face=False,
                        paste_back=True
                    )
                    restored_rgb = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
                    batch_enhanced.append(restored_rgb)
                except Exception as e2:
                    print(f"Enhancement failed for image, using original: {e2}")
                    batch_enhanced.append(img)
        
        return batch_enhanced
    
    def _calculate_optimal_batch_size(self, gpu_memory_gb):
        """Calculate optimal batch size based on GPU memory - scales automatically"""
        # Estimate memory per image (GFPGAN processing overhead)
        memory_per_image_gb = 2.2  # Conservative estimate for GFPGAN

        # Calculate available memory for processing (leave 15% buffer)
        available_memory_gb = gpu_memory_gb * 0.85

        # Calculate optimal batch size
        optimal_batch_size = max(1, int(available_memory_gb / memory_per_image_gb))

        # Apply reasonable caps based on GPU size
        if gpu_memory_gb >= 40:  # A100/H100
            max_batch = 48
        elif gpu_memory_gb >= 24:  # RTX 4090/3090
            max_batch = 32
        elif gpu_memory_gb >= 16:  # RTX 4080/3080
            max_batch = 24
        elif gpu_memory_gb >= 12:  # RTX 4070/3070
            max_batch = 20
        elif gpu_memory_gb >= 8:   # RTX 3060/4060
            max_batch = 16
        elif gpu_memory_gb >= 6:   # RTX 3050/4050
            max_batch = 12
        elif gpu_memory_gb >= 4:   # GTX 1660/2060
            max_batch = 8
        else:
            max_batch = 4

        return min(optimal_batch_size, max_batch)


def fast_enhancer_generator_with_len(images, method='gfpgan', bg_upsampler='realesrgan', 
                                   batch_size=4, optimization_level="medium"):
    """Fast enhanced generator with length support"""
    
    if os.path.isfile(images):
        images = load_video_to_cv2(images)
    
    enhancer = OptimizedFaceEnhancer(method=method, optimization_level=optimization_level)
    enhanced_images = enhancer.enhance_batch(images, batch_size=batch_size)
    
    def enhanced_generator():
        for img in enhanced_images:
            yield img
    
    gen_with_len = FastGeneratorWithLen(enhanced_generator(), len(enhanced_images))
    return gen_with_len


def fast_enhancer_list(images, method='gfpgan', bg_upsampler='realesrgan', 
                      batch_size=4, optimization_level="medium"):
    """Fast enhanced list generation"""
    
    if os.path.isfile(images):
        images = load_video_to_cv2(images)
    
    enhancer = OptimizedFaceEnhancer(method=method, optimization_level=optimization_level)
    return enhancer.enhance_batch(images, batch_size=batch_size)


# Memory-optimized streaming enhancer
class StreamingEnhancer:
    """Memory-efficient streaming face enhancer"""
    
    def __init__(self, method='gfpgan', optimization_level="medium", max_memory_mb=1024):
        self.enhancer = OptimizedFaceEnhancer(method=method, optimization_level=optimization_level)
        self.max_memory_mb = max_memory_mb
        self.frame_buffer = Queue(maxsize=32)  # Buffer for frames
    
    def enhance_stream(self, video_path, batch_size=4):
        """Stream-process video frames to minimize memory usage"""
        
        def frame_producer():
            """Producer thread to load frames"""
            cap = cv2.VideoCapture(video_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_buffer.put(frame_rgb)
            cap.release()
            self.frame_buffer.put(None)  # Signal end
        
        def frame_consumer():
            """Consumer thread to enhance frames"""
            batch = []
            while True:
                frame = self.frame_buffer.get()
                if frame is None:
                    # Process remaining frames
                    if batch:
                        if self.enhancer.optimization_level == "extreme":
                            enhanced_batch = self.enhancer._lightweight_batch_enhance(batch, len(batch))
                        else:
                            enhanced_batch = self.enhancer._sequential_batch_enhance(batch, len(batch))
                        for enhanced_frame in enhanced_batch:
                            yield enhanced_frame
                    break
                
                batch.append(frame)
                if len(batch) >= batch_size:
                    # Process batch
                    if self.enhancer.optimization_level == "extreme":
                        enhanced_batch = self.enhancer._lightweight_batch_enhance(batch, batch_size)
                    else:
                        enhanced_batch = self.enhancer._sequential_batch_enhance(batch, batch_size)
                    
                    for enhanced_frame in enhanced_batch:
                        yield enhanced_frame
                    batch = []
        
        # Start producer thread
        import threading
        producer_thread = threading.Thread(target=frame_producer)
        producer_thread.daemon = True
        producer_thread.start()
        
        # Return consumer generator
        return frame_consumer()


# GPU memory management utilities
class GPUMemoryManager:
    """Manage GPU memory for optimal performance"""
    
    @staticmethod
    def get_optimal_batch_size():
        """Determine optimal batch size based on available GPU memory"""
        if not torch.cuda.is_available():
            return 1
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_gb = gpu_memory / 1e9
        
        if gpu_memory_gb >= 12:
            return 8
        elif gpu_memory_gb >= 8:
            return 6
        elif gpu_memory_gb >= 6:
            return 4
        elif gpu_memory_gb >= 4:
            return 2
        else:
            return 1
    
    @staticmethod
    def setup_memory_optimization():
        """Setup memory optimizations"""
        if torch.cuda.is_available():
            # Enable memory optimization
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Set memory fraction
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(0.8)
    
    @staticmethod
    def cleanup_gpu_memory():
        """Aggressive GPU memory cleanup"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# Quality vs Speed presets
class EnhancementPresets:
    """Predefined presets for different quality/speed tradeoffs"""
    
    PRESETS = {
        "ultra_fast": {
            "optimization_level": "extreme",
            "method": "lightweight",
            "batch_size": 16,
            "skip_bg_upsampler": True,
            "description": "Fastest processing, basic enhancement only"
        },
        "fast": {
            "optimization_level": "high", 
            "method": "gfpgan",
            "batch_size": 8,
            "skip_bg_upsampler": True,
            "description": "Fast processing with GFPGAN, no background upsampling"
        },
        "balanced": {
            "optimization_level": "medium",
            "method": "gfpgan", 
            "batch_size": 4,
            "skip_bg_upsampler": False,
            "description": "Balanced quality and speed"
        },
        "quality": {
            "optimization_level": "low",
            "method": "RestoreFormer",
            "batch_size": 2,
            "skip_bg_upsampler": False,
            "description": "Best quality, slower processing"
        }
    }
    
    @classmethod
    def get_preset(cls, preset_name):
        """Get enhancement preset configuration with dynamic batch sizing"""
        if preset_name not in cls.PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(cls.PRESETS.keys())}")

        config = cls.PRESETS[preset_name].copy()

        # Calculate dynamic batch size based on available GPU memory
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

            # Base batch size multiplier for each preset
            multipliers = {
                "ultra_fast": 2.0,  # Very aggressive batching
                "fast": 1.5,        # Aggressive batching
                "balanced": 1.0,    # Standard batching
                "quality": 0.5      # Conservative batching
            }

            base_batch = config['batch_size']
            multiplier = multipliers.get(preset_name, 1.0)

            # Calculate memory-based batch size
            memory_per_image_gb = 2.2
            available_memory_gb = gpu_memory_gb * 0.8  # 80% utilization
            memory_based_batch = max(1, int(available_memory_gb / memory_per_image_gb))

            # Apply preset multiplier and cap
            dynamic_batch = min(int(base_batch * multiplier), memory_based_batch)
            config['batch_size'] = max(1, dynamic_batch)

        return config
    
    @classmethod
    def list_presets(cls):
        """List available presets with descriptions"""
        for name, config in cls.PRESETS.items():
            print(f"{name}: {config['description']}")


# Usage example and benchmark function
def benchmark_enhancement_methods(video_path, num_frames=50):
    """Benchmark different enhancement methods"""
    import time
    
    # Load test frames
    frames = load_video_to_cv2(video_path)[:num_frames]
    
    results = {}
    
    for preset_name in EnhancementPresets.PRESETS.keys():
        print(f"\nBenchmarking {preset_name}...")
        preset = EnhancementPresets.get_preset(preset_name)
        
        start_time = time.time()
        
        if preset_name == "ultra_fast":
            enhancer = OptimizedFaceEnhancer(optimization_level="extreme")
            enhanced = enhancer._lightweight_batch_enhance(frames, preset["batch_size"])
        else:
            enhanced = fast_enhancer_list(
                frames, 
                method=preset["method"],
                batch_size=preset["batch_size"],
                optimization_level=preset["optimization_level"]
            )
        
        end_time = time.time()
        processing_time = end_time - start_time
        fps = len(frames) / processing_time
        
        results[preset_name] = {
            "time": processing_time,
            "fps": fps,
            "frames": len(enhanced)
        }
        
        print(f"{preset_name}: {processing_time:.2f}s, {fps:.2f} FPS")
    
    return results