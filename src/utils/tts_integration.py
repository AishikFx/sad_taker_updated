"""
SadTalker TTS Integration Module
Converts text to speech using Google TTS with Indian accent for female voice.
"""

import asyncio
import tempfile
import logging
import os
import importlib
from typing import Optional, Tuple
from io import BytesIO

# Google TTS import
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

# Audio processing import
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


class SadTalkerTTS:
    """
    Text-to-Speech integration for SadTalker with gender support.
    Currently supports female voice with Indian accent using Google TTS.
    """
    
    def __init__(self):
        self.supported_genders = ['female']
        self.logger = logging.getLogger(__name__)
        
        # Check dependencies
        if not GTTS_AVAILABLE:
            raise ImportError("gtts is required. Install with: pip install gtts")
        
        if not PYDUB_AVAILABLE:
            raise ImportError("pydub is required. Install with: pip install pydub")
    
    def validate_gender(self, gender: str) -> bool:
        """
        Validate if the requested gender is supported.
        
        Args:
            gender: Gender string ('male' or 'female')
            
        Returns:
            bool: True if supported, False otherwise
        """
        return gender.lower() in self.supported_genders
    
    def generate_speech_from_text(
        self, 
        text: str, 
        gender: str = 'female',
        output_path: Optional[str] = None
    ) -> str:
        """
        Convert text to speech and save as audio file.
        
        Args:
            text: Input text to convert to speech
            gender: Gender of voice ('female' only supported)
            output_path: Optional path to save audio file. If None, creates temp file.
            
        Returns:
            str: Path to generated audio file
            
        Raises:
            ValueError: If gender is not supported or text is empty
            RuntimeError: If TTS generation fails
        """
        # Validate inputs
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        if not self.validate_gender(gender):
            raise ValueError(f"Gender '{gender}' not supported. Only {self.supported_genders} are supported.")
        
        text = text.strip()
        self.logger.info(f"Generating speech for text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        try:
            # Create gTTS object with Indian accent
            tts = gTTS(
                text=text,
                lang='en',
                tld='co.in',  # Indian accent
                slow=False
            )
            
            # Determine output path
            if output_path is None:
                # Create temporary file
                temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
                output_path = temp_file.name
                temp_file.close()
            
            # Generate speech
            tts.save(output_path)
            self.logger.info(f"Speech generated successfully: {output_path}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"TTS generation failed: {e}")
            raise RuntimeError(f"Failed to generate speech: {str(e)}")
    
    def convert_to_wav(self, mp3_path: str, wav_path: Optional[str] = None) -> str:
        """
        Convert MP3 file to WAV format (required by SadTalker).
        
        Args:
            mp3_path: Path to input MP3 file
            wav_path: Optional path for output WAV file. If None, creates temp file.
            
        Returns:
            str: Path to WAV file
        """
        try:
            # Load MP3 file
            audio = AudioSegment.from_mp3(mp3_path)
            
            # Determine output path
            if wav_path is None:
                # Create temporary WAV file
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                wav_path = temp_file.name
                temp_file.close()
            
            # Export as WAV
            audio.export(wav_path, format="wav")
            self.logger.info(f"Converted to WAV: {wav_path}")
            
            return wav_path
            
        except Exception as e:
            self.logger.error(f"WAV conversion failed: {e}")
            raise RuntimeError(f"Failed to convert to WAV: {str(e)}")
    
    def text_to_audio_for_sadtalker(
        self, 
        text: str, 
        gender: str = 'female',
        cleanup_temp: bool = True
    ) -> str:
        """
        Complete pipeline: Convert text to WAV audio file ready for SadTalker.
        
        Args:
            text: Input text to convert to speech
            gender: Gender of voice ('female' only supported)
            cleanup_temp: Whether to cleanup temporary MP3 file
            
        Returns:
            str: Path to WAV audio file ready for SadTalker
        """
        # Step 1: Generate speech as MP3
        mp3_path = self.generate_speech_from_text(text, gender)
        
        try:
            # Step 2: Convert to WAV
            wav_path = self.convert_to_wav(mp3_path)
            
            # Step 3: Cleanup temporary MP3 if requested
            if cleanup_temp:
                try:
                    os.unlink(mp3_path)
                    self.logger.info(f"Cleaned up temporary MP3: {mp3_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup MP3: {e}")
            
            return wav_path
            
        except Exception as e:
            # Cleanup MP3 on failure
            try:
                os.unlink(mp3_path)
            except:
                pass
            raise e


# Async wrapper for non-blocking TTS generation
class AsyncSadTalkerTTS:
    """
    Async wrapper for SadTalkerTTS to avoid blocking the main thread.
    """
    
    def __init__(self):
        self.tts = SadTalkerTTS()
    
    async def text_to_audio_async(
        self, 
        text: str, 
        gender: str = 'female',
        cleanup_temp: bool = True
    ) -> str:
        """
        Async version of text_to_audio_for_sadtalker.
        """
        loop = asyncio.get_event_loop()
        
        # Run TTS in thread pool to avoid blocking
        wav_path = await loop.run_in_executor(
            None, 
            self.tts.text_to_audio_for_sadtalker,
            text,
            gender,
            cleanup_temp
        )
        
        return wav_path


# Convenience functions
def text_to_speech_for_sadtalker(text: str, gender: str = 'female') -> str:
    """
    Convenience function to convert text to speech for SadTalker.
    
    Args:
        text: Input text
        gender: Voice gender ('female' only)
        
    Returns:
        str: Path to WAV file ready for SadTalker
    """
    tts = SadTalkerTTS()
    return tts.text_to_audio_for_sadtalker(text, gender)


async def async_text_to_speech_for_sadtalker(text: str, gender: str = 'female') -> str:
    """
    Async convenience function to convert text to speech for SadTalker.
    """
    tts = AsyncSadTalkerTTS()
    return await tts.text_to_audio_async(text, gender)


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    def test_tts():
        """Test the TTS functionality."""
        print("Testing SadTalker TTS Integration...")
        
        # Test text
        test_text = "Hello! I am a virtual avatar created using SadTalker with text-to-speech. This is a demonstration of the Indian accent female voice."
        
        try:
            # Test synchronous version
            print("\n1. Testing synchronous TTS...")
            tts = SadTalkerTTS()
            
            # Test gender validation
            print("   - Testing gender validation...")
            assert tts.validate_gender('female') == True
            assert tts.validate_gender('male') == False
            print("   ✓ Gender validation working")
            
            # Test audio generation
            print("   - Generating speech...")
            wav_path = tts.text_to_audio_for_sadtalker(test_text, 'female')
            print(f"   ✓ Audio generated: {wav_path}")
            
            # Check file exists and has content
            if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                print(f"   ✓ WAV file created successfully ({os.path.getsize(wav_path)} bytes)")
            else:
                print("   ✗ WAV file creation failed")
            
            # Test error cases
            print("   - Testing error cases...")
            try:
                tts.text_to_audio_for_sadtalker("", 'female')
                print("   ✗ Empty text should raise error")
            except ValueError:
                print("   ✓ Empty text error handled")
            
            try:
                tts.text_to_audio_for_sadtalker("Test", 'male')
                print("   ✗ Male gender should raise error")
            except ValueError:
                print("   ✓ Male gender error handled")
            
            # Cleanup
            try:
                os.unlink(wav_path)
                print("   ✓ Cleanup completed")
            except:
                pass
                
        except Exception as e:
            print(f"   ✗ TTS test failed: {e}")
            return False
        
        return True
    
    async def test_async_tts():
        """Test the async TTS functionality."""
        print("\n2. Testing asynchronous TTS...")
        
        test_text = "This is an asynchronous test of the text-to-speech functionality with Indian accent."
        
        try:
            wav_path = await async_text_to_speech_for_sadtalker(test_text, 'female')
            print(f"   ✓ Async audio generated: {wav_path}")
            
            # Cleanup
            try:
                os.unlink(wav_path)
                print("   ✓ Async cleanup completed")
            except:
                pass
                
            return True
            
        except Exception as e:
            print(f"   ✗ Async TTS test failed: {e}")
            return False
    
    # Run tests
    success = test_tts()
    if success:
        # Run async test
        try:
            success = asyncio.run(test_async_tts())
        except Exception as e:
            print(f"Async test failed: {e}")
            success = False
    
    if success:
        print("\n🎉 All TTS tests passed! Ready for SadTalker integration.")
    else:
        print("\n❌ TTS tests failed. Please check dependencies:")
        print("   pip install gtts pydub")