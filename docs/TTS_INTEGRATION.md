# SadTalker Text-to-Speech Integration

This document explains how to use the new Text-to-Speech (TTS) functionality in SadTalker, which allows you to generate talking head videos from text input instead of audio files.

## Overview

The TTS integration adds the ability to:
- Convert text to speech using Google TTS with Indian accent
- Generate videos using text input instead of audio files
- Support for female voice only (male voice support coming soon)
- API endpoints for programmatic access
- Simple command-line usage

## Installation

### Install TTS Dependencies

```bash
pip install -r requirements_tts.txt
```

Or install manually:
```bash
pip install gtts pydub fastapi uvicorn python-multipart
```

### Verify Installation

Test the TTS module:
```bash
python src/utils/tts_integration.py
```

## Usage Methods

### 1. Command Line Interface

Generate video using text input:

```bash
python inference.py \
    --source_image "./examples/source_image/full_body_1.png" \
    --input_text "Hello! I am a virtual avatar created using SadTalker with text-to-speech." \
    --gender female \
    --result_dir "./results" \
    --size 256
```

**Parameters:**
- `--input_text`: Text to convert to speech (required)
- `--gender`: Voice gender ("female" only supported)
- `--source_image`: Path to source image
- Other parameters work the same as before

**Note:** When using `--input_text`, do not provide `--driven_audio`. The system will automatically generate audio from text.

### 2. API Server

Start the TTS API server:

```bash
python tts_api.py
```

The server will start on `http://localhost:8000` with the following endpoints:

#### Generate Video from Text (Simple)

```bash
curl -X POST "http://localhost:8000/generate-video-simple" \
     -F "image=@./examples/source_image/full_body_1.png" \
     -F "text=Hello! This is a test of SadTalker TTS integration." \
     -F "gender=female"
```

#### Generate Video from Text (Advanced)

```bash
curl -X POST "http://localhost:8000/generate-video" \
     -F "image=@./examples/source_image/full_body_1.png" \
     -F "text=Hello! This is a test with enhanced quality." \
     -F "gender=female" \
     -F "enhancer=gfpgan" \
     -F "size=512" \
     -F "expression_scale=1.2"
```

#### Health Check

```bash
curl http://localhost:8000/health
```

#### Supported Genders

```bash
curl http://localhost:8000/supported-genders
```

### 3. Python Code Integration

```python
from src.utils.tts_integration import SadTalkerTTS
from inference import main

# Initialize TTS
tts = SadTalkerTTS()

# Generate audio from text
audio_path = tts.text_to_audio_for_sadtalker(
    text="Hello! I am testing SadTalker TTS integration.",
    gender="female"
)

# Create arguments for SadTalker
class Args:
    def __init__(self):
        self.source_image = "./examples/source_image/full_body_1.png"
        self.driven_audio = audio_path
        self.input_text = "Hello! I am testing SadTalker TTS integration."
        self.gender = "female"
        # ... other parameters

args = Args()
output_video = main(args)
print(f"Video generated: {output_video}")
```

## API Documentation

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/supported-genders` | GET | List supported voice genders |
| `/generate-video` | POST | Generate video with full options |
| `/generate-video-simple` | POST | Generate video with basic options |

### Request Parameters

#### `/generate-video` (POST)

**Form Data:**
- `image` (file): Source image file (required)
- `text` (string): Text to convert to speech (required)
- `gender` (string): Voice gender ("female" only, default: "female")
- `enhancer` (string): Face enhancer ("gfpgan", "RestoreFormer", optional)
- `background_enhancer` (string): Background enhancer ("realesrgan", optional)
- `size` (integer): Output size (default: 256)
- `expression_scale` (float): Expression scale (default: 1.0)
- `pose_style` (integer): Pose style 0-46 (default: 0)

**Response:**
```json
{
    "success": true,
    "message": "Video generated successfully from text",
    "video_path": "/path/to/generated/video.mp4",
    "session_id": "uuid-string",
    "processing_time": 15.42
}
```

### Error Responses

```json
{
    "success": false,
    "error": "Only 'female' gender is currently supported for TTS",
    "status_code": 400
}
```

## Testing

### Run All Tests

```bash
python test_tts_integration.py
```

### Test Direct Inference Only

```bash
python test_tts_integration.py --test-type direct
```

### Test API Server Only

```bash
python test_tts_integration.py --test-type api
```

## Voice Options

### Supported Genders

Currently supported:
- **female**: Indian accent using Google TTS

### Coming Soon

- **male**: Male voice (currently throws error)
- Multiple accents and languages
- Voice speed control
- Emotion control

## Performance Considerations

### TTS Generation Time

- Text-to-speech: ~1-3 seconds
- Audio processing: Same as before
- Video generation: Same as before
- **Total overhead**: ~1-3 seconds additional

### Memory Usage

- TTS uses minimal additional memory
- Audio files are temporary and cleaned up automatically
- Same GPU memory requirements as before

### Optimization Tips

1. **Shorter Text**: Break long text into multiple videos
2. **Batch Processing**: Process multiple texts in sequence
3. **Caching**: Cache generated audio for repeated text
4. **GPU**: Use GPU for faster video generation

## Troubleshooting

### Common Issues

#### 1. "gtts not found" Error

```bash
pip install gtts
```

#### 2. "pydub not found" Error

```bash
pip install pydub
```

For audio format support:
```bash
# On Windows
pip install pydub[soundfile]

# On Linux/Mac
sudo apt-get install ffmpeg  # or brew install ffmpeg
```

#### 3. "Only female gender supported" Error

This is expected behavior. Male voice support is coming soon.

#### 4. API Server Won't Start

Check if port 8000 is available:
```bash
# Change port in tts_api.py if needed
uvicorn tts_api:app --host 0.0.0.0 --port 8001
```

#### 5. TTS Audio Quality Issues

- Check internet connection (Google TTS requires internet)
- Try shorter text segments
- Verify text encoding (UTF-8 recommended)

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run your TTS command
```

### Log Files

TTS operations are logged to console. For persistent logs:

```bash
python inference.py --input_text "Your text" --gender female 2>&1 | tee tts_log.txt
```

## Examples

### Example 1: Basic Usage

```bash
python inference.py \
    --source_image "./examples/source_image/art_0.png" \
    --input_text "Welcome to SadTalker! This is a demonstration of text-to-speech integration." \
    --gender female
```

### Example 2: High Quality with Enhancement

```bash
python inference.py \
    --source_image "./examples/source_image/full_body_1.png" \
    --input_text "This video uses enhanced quality settings for better results." \
    --gender female \
    --enhancer gfpgan \
    --size 512 \
    --expression_scale 1.2
```

### Example 3: API Usage with Python

```python
import requests

url = "http://localhost:8000/generate-video-simple"

with open("./examples/source_image/art_0.png", "rb") as image_file:
    files = {"image": image_file}
    data = {
        "text": "Hello from the API! This is a test of SadTalker TTS integration.",
        "gender": "female"
    }
    
    response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Video generated: {result['video_path']}")
    else:
        print(f"Error: {response.text}")
```

### Example 4: JavaScript/Frontend Integration

```javascript
const formData = new FormData();
formData.append('image', imageFile);
formData.append('text', 'Hello from JavaScript! This is SadTalker TTS integration.');
formData.append('gender', 'female');

fetch('http://localhost:8000/generate-video-simple', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    if (data.success) {
        console.log('Video generated:', data.video_path);
    } else {
        console.error('Error:', data.error);
    }
});
```

## Integration with Existing Workflows

### With Original Audio Files

You can still use audio files as before:

```bash
python inference.py \
    --source_image "./examples/source_image/full_body_1.png" \
    --driven_audio "./examples/driven_audio/bus_chinese.wav"
```

### Mixed Usage

Process both text and audio in the same script:

```python
# Process text
output1 = main(args_with_text)

# Process audio file  
output2 = main(args_with_audio)
```

## Deployment

### Docker Support

Add to your Dockerfile:

```dockerfile
# Install TTS dependencies
RUN pip install gtts pydub fastapi uvicorn python-multipart

# Copy TTS files
COPY src/utils/tts_integration.py /app/src/utils/
COPY tts_api.py /app/
COPY requirements_tts.txt /app/
```

### Production Considerations

1. **Load Balancing**: Run multiple API instances
2. **Caching**: Implement Redis caching for repeated text
3. **Rate Limiting**: Add rate limiting for API endpoints
4. **Monitoring**: Add health checks and metrics
5. **Security**: Add authentication for production use

## Roadmap

### Coming Soon

- [ ] Male voice support
- [ ] Multiple language support
- [ ] Voice emotion control
- [ ] Batch text processing
- [ ] Real-time streaming
- [ ] Voice cloning integration

### Planned Features

- [ ] SSML support for advanced speech control
- [ ] Custom voice training
- [ ] Lip-sync accuracy improvements
- [ ] Background music integration
- [ ] Subtitle generation

## Support

For issues and questions:

1. Check this documentation
2. Run the test script: `python test_tts_integration.py`
3. Check logs for detailed error messages
4. Create an issue with:
   - Error message
   - Input text and parameters
   - System information (OS, GPU, etc.)

## License

The TTS integration follows the same license as SadTalker.