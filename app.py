import os
from dotenv import load_dotenv
import openai
import asyncio
import json
import base64
from concurrent.futures import ThreadPoolExecutor
from aiohttp import web, WSMsgType
import aiohttp_cors
import logging
import io
import tempfile
import subprocess
import speech_recognition as sr
from gtts import gTTS
import cv2
import numpy as np

# Try to import FER, but make it optional
try:
    from fer import FER
    HAS_EMOTION_DETECTION = True
except ImportError:
    HAS_EMOTION_DETECTION = False
    logging.warning("FER library not available. Emotion detection will be disabled.")

# --- CONFIGURATION AND INITIALIZATION ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- API & Server Config ---
PORT = int(os.getenv('PORT', 8080))
HOST = os.getenv('HOST', '0.0.0.0')
HISTORY_FILE = "conversation_history.json"

# --- API CLIENTS & ML MODELS ---
try:
    openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    if HAS_EMOTION_DETECTION:
        emotion_detector = FER(mtcnn=True)
        logger.info("✅ Emotion detection enabled.")
    else:
        emotion_detector = None
        logger.info("⚠️ Emotion detection disabled.")
    
    recognizer = sr.Recognizer()
    logger.info("✅ API clients and speech recognizer initialized.")
except Exception as e:
    logger.error(f"❌ Failed to initialize clients: {e}")
    raise

# --- GLOBAL STATE & CONTEXT MANAGEMENT ---
active_connections = {}
conversation_context = []
visual_context = {"description": "No visual data available.", "emotion": "unknown"}

executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

# --- CONVERSATION PERSISTENCE ---
def load_conversation_history():
    global conversation_context
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                conversation_context = json.load(f)
            logger.info(f"🧠 Conversation history loaded.")
        except Exception as e:
            logger.error(f"⚠️ Could not load history: {e}. Starting fresh.")
            conversation_context = []

def save_conversation_history():
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(conversation_context, f, indent=2)
    except Exception as e:
        logger.error(f"⚠️ Could not save history: {e}.")

# --- AUDIO PROCESSING FUNCTIONS ---
async def convert_webm_to_wav(webm_bytes):
    """Convert WebM audio to WAV format using ffmpeg."""
    try:
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as webm_file:
            webm_file.write(webm_bytes)
            webm_path = webm_file.name

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as wav_file:
            wav_path = wav_file.name

        # Convert using ffmpeg
        cmd = [
            'ffmpeg', '-i', webm_path, 
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',      # mono
            '-y',            # overwrite output
            wav_path
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()
        
        # Read the converted WAV file
        with open(wav_path, 'rb') as f:
            wav_bytes = f.read()
        
        # Cleanup
        os.unlink(webm_path)
        os.unlink(wav_path)
        
        return wav_bytes
        
    except Exception as e:
        logger.error(f"❌ Audio conversion error: {e}")
        return None

async def transcribe_audio_on_server(audio_bytes):
    """Transcribes audio using the speech_recognition library on the backend."""
    loop = asyncio.get_event_loop()
    try:
        # Convert WebM to WAV first
        wav_bytes = await convert_webm_to_wav(audio_bytes)
        if not wav_bytes:
            return ""
        
        # Use speech_recognition with the converted audio
        with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
            audio_data = await loop.run_in_executor(executor, recognizer.record, source)
        
        # Use Google's free web speech API for transcription
        text = await loop.run_in_executor(executor, recognizer.recognize_google, audio_data)
        logger.info(f"🎤 Transcription: '{text}'")
        return text
        
    except sr.UnknownValueError:
        logger.warning("👂 Google Speech Recognition could not understand audio.")
        return ""
    except sr.RequestError as e:
        logger.error(f"❌ Could not request results from Google Speech Recognition service; {e}")
        return ""
    except Exception as e:
        logger.error(f"❌ Transcription error: {e}")
        return ""

# --- CORE AI FUNCTIONS ---
async def get_text_response(user_input):
    """Generates a text response from GPT-4o."""
    global conversation_context, visual_context
    
    logger.info(f"🧠 Processing: '{user_input[:50]}...'")
    
    emotion_status = "enabled" if HAS_EMOTION_DETECTION else "disabled"
    system_prompt = f"""You are Nexus, an emotionally intelligent AI companion. You are friendly, casual, and empathetic.
Your sensory context is: Visuals: {visual_context['description']}. Detected Emotion: {visual_context['emotion']}.
Emotion detection: {emotion_status}.
Use this awareness to respond naturally and keep responses conversational and under 150 words.
"""
    
    conversation_context.append({"role": "user", "content": user_input})
    messages = [{"role": "system", "content": system_prompt}] + conversation_context[-12:]
    
    try:
        response = await asyncio.to_thread(
            openai_client.chat.completions.create, 
            model="gpt-4o", 
            messages=messages, 
            max_tokens=150,
            temperature=0.7
        )
        ai_response_text = response.choices[0].message.content.strip()
        conversation_context.append({"role": "assistant", "content": ai_response_text})
        save_conversation_history()
        logger.info(f"🤖 AI Response: '{ai_response_text}'")
        return ai_response_text
    except Exception as e:
        logger.error(f"❌ OpenAI error: {e}")
        return "I'm having a little trouble connecting to my thoughts right now."

async def stream_audio_response(text, ws):
    """Generates audio with gTTS and streams it to the client."""
    logger.info("🔊 Generating audio with gTTS...")
    try:
        # Create an in-memory binary stream for the audio
        mp3_fp = io.BytesIO()
        tts = await asyncio.to_thread(gTTS, text=text, lang='en', slow=False)
        await asyncio.to_thread(tts.write_to_fp, mp3_fp)
        mp3_fp.seek(0)
        
        # Stream the audio in chunks
        chunk_size = 4096
        while True:
            chunk = mp3_fp.read(chunk_size)
            if not chunk:
                break
            await ws.send_str(json.dumps({
                "type": "audio_chunk",
                "data": base64.b64encode(chunk).decode('utf-8')
            }))
        
        await ws.send_str(json.dumps({"type": "audio_stop"}))
        logger.info("🔊 Audio streaming completed.")
        
    except Exception as e:
        logger.error(f"❌ gTTS streaming error: {e}")
        await ws.send_str(json.dumps({"type": "error", "message": "Audio generation failed"}))

async def analyze_image_with_emotion(image_data):
    """Analyzes an image for visual content and detects facial emotions."""
    global visual_context
    
    try:
        img_bytes = base64.b64decode(image_data)
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        if img is None:
            logger.warning("⚠️ Could not decode image data")
            return

        # Emotion detection if available
        if HAS_EMOTION_DETECTION and emotion_detector:
            try:
                emotion_results = await asyncio.to_thread(emotion_detector.detect_emotions, img)
                
                if emotion_results and len(emotion_results) > 0:
                    emotions = emotion_results[0]['emotions']
                    top_emotion = max(emotions, key=emotions.get)
                    confidence = emotions[top_emotion]
                    
                    if confidence > 0.3:  # Only use if confidence is reasonable
                        visual_context['emotion'] = top_emotion
                        logger.info(f"😊 Detected emotion: {top_emotion} ({confidence:.2f})")
                    else:
                        visual_context['emotion'] = "neutral"
                else:
                    visual_context['emotion'] = "neutral"
                    
            except Exception as e:
                logger.error(f"❌ Emotion detection error: {e}")
                visual_context['emotion'] = "unknown"
        
        # Basic visual analysis
        height, width = img.shape[:2]
        visual_context['description'] = f"Camera active - {width}x{height} image"
        
        # TODO: Add GPT-4o vision analysis here if needed
        # This would require sending the image to OpenAI's vision API
        
    except Exception as e:
        logger.error(f"❌ Image analysis error: {e}")
        visual_context = {"description": "Image analysis failed", "emotion": "unknown"}

# --- WEBSOCKET HANDLER ---
async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    client_addr = request.remote
    active_connections[ws] = client_addr
    logger.info(f"🌌 Connection established: {client_addr}")
    
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    msg_type = data.get('type')

                    if msg_type == 'audio_input':
                        audio_b64 = data.get('data', '')
                        
                        # Handle data URL format (data:audio/webm;base64,...)
                        if ',' in audio_b64:
                            audio_b64 = audio_b64.split(',')[1]
                        
                        audio_bytes = base64.b64decode(audio_b64)
                        
                        if len(audio_bytes) > 0:
                            await ws.send_str(json.dumps({"type": "status", "message": "processing"}))
                            transcription = await transcribe_audio_on_server(audio_bytes)

                            if transcription.strip():
                                await ws.send_str(json.dumps({"type": "status", "message": "thinking"}))
                                ai_text = await get_text_response(transcription)
                                await stream_audio_response(ai_text, ws)
                            else:
                                await ws.send_str(json.dumps({"type": "status", "message": "listening"}))
                        else:
                            logger.warning("⚠️ Received empty audio data")

                    elif msg_type == 'camera_frame':
                        frame_data = data.get('data', '')
                        if frame_data:
                            await analyze_image_with_emotion(frame_data)

                except json.JSONDecodeError:
                    logger.error("❌ Invalid JSON received")
                except Exception as e:
                    logger.error(f"❌ WebSocket message handling error: {e}")
                    
            elif msg.type == WSMsgType.ERROR:
                logger.error(f'❌ WebSocket error: {ws.exception()}')

    except Exception as e:
        logger.error(f"❌ WebSocket handler error: {e}")
    finally:
        if ws in active_connections:
            del active_connections[ws]
        logger.info(f"🔌 Connection closed: {client_addr}")
    
    return ws

# --- HTML FRONTEND ---
async def serve_cosmic_interface(request):
    """Serves the main HTML/CSS/JS interface for backend audio processing."""
    emotion_status = "ENABLED" if HAS_EMOTION_DETECTION else "DISABLED"
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NEXUS</title>
    <style>
        body {{ 
            font-family: 'Courier New', monospace; 
            background: linear-gradient(135deg, #0a0a0a, #1a1a1a); 
            color: #fff; 
            display: flex; 
            flex-direction: column; 
            justify-content: center; 
            align-items: center; 
            height: 100vh; 
            margin: 0; 
            overflow: hidden;
        }}
        .status {{ 
            font-size: 2.5rem; 
            text-align: center;
            text-shadow: 0 0 20px currentColor;
            transition: all 0.3s ease;
        }}
        .info {{ 
            font-size: 1rem; 
            margin-top: 20px; 
            opacity: 0.7; 
            text-align: center;
        }}
        .instructions {{
            position: fixed;
            bottom: 20px;
            font-size: 0.9rem;
            opacity: 0.5;
            text-align: center;
        }}
        
        .state-listening {{ color: #00D4FF; }} 
        .state-user-speaking {{ color: #00F5FF; }}
        .state-ai-speaking {{ color: #FF006E; }} 
        .state-thinking {{ color: #8338EC; }}
        .state-processing {{ color: #FFB800; }}
        .state-error {{ color: #FF4444; }}
        
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.6; }}
        }}
        
        .state-user-speaking .status,
        .state-thinking .status,
        .state-processing .status {{
            animation: pulse 1.5s infinite;
        }}
    </style>
</head>
<body class="state-initializing">
    <div class="status" id="statusText">INITIALIZING...</div>
    <div class="info">
        <div>Emotion Detection: {emotion_status}</div>
        <div id="connectionInfo">Connecting...</div>
    </div>
    <div class="instructions">
        Click anywhere to start/stop recording
    </div>
    
    <script>
        let ws, mediaRecorder, reconnectAttempts = 0;
        let isAISpeaking = false, isRecording = false;
        let audioQueue = [], isPlayingFromQueue = false;

        const statusText = document.getElementById('statusText');
        const connectionInfo = document.getElementById('connectionInfo');

        function updateState(state, text) {{
            document.body.className = `state-${{state}}`;
            statusText.textContent = text;
        }}

        async function playAudioFromQueue() {{
            if (isPlayingFromQueue || audioQueue.length === 0) return;
            isPlayingFromQueue = true;
            updateState('ai-speaking', 'SPEAKING...');
            isAISpeaking = true;
            
            const data = audioQueue.shift();
            try {{
                const audioData = atob(data);
                const buffer = new Uint8Array(audioData.length);
                for (let i = 0; i < audioData.length; i++) {{
                    buffer[i] = audioData.charCodeAt(i);
                }}
                
                const blob = new Blob([buffer], {{ type: 'audio/mpeg' }});
                const url = URL.createObjectURL(blob);
                const audio = new Audio(url);
                
                audio.onended = () => {{
                    URL.revokeObjectURL(url);
                    isPlayingFromQueue = false;
                    setTimeout(playAudioFromQueue, 100);
                }};
                
                audio.onerror = (e) => {{
                    console.error("Audio playback error:", e);
                    URL.revokeObjectURL(url);
                    isPlayingFromQueue = false;
                    setTimeout(playAudioFromQueue, 100);
                }};
                
                await audio.play();
                
            }} catch (e) {{
                console.error("Audio processing error:", e);
                isPlayingFromQueue = false;
                setTimeout(playAudioFromQueue, 100);
            }}
        }}

        async function initMicrophone() {{
            try {{
                const stream = await navigator.mediaDevices.getUserMedia({{ 
                    audio: {{ 
                        echoCancellation: true,
                        noiseSuppression: true,
                        sampleRate: 16000
                    }} 
                }});
                
                mediaRecorder = new MediaRecorder(stream, {{ 
                    mimeType: 'audio/webm;codecs=opus' 
                }});
                
                mediaRecorder.onstart = () => {{
                    isRecording = true;
                    console.log("Recording started");
                }};
                
                mediaRecorder.onstop = () => {{
                    isRecording = false;
                    console.log("Recording stopped");
                }};

                mediaRecorder.ondataavailable = (event) => {{
                    if (event.data.size > 0 && ws && ws.readyState === WebSocket.OPEN) {{
                        const reader = new FileReader();
                        reader.onloadend = () => {{
                            ws.send(JSON.stringify({{ 
                                type: 'audio_input', 
                                data: reader.result 
                            }}));
                        }};
                        reader.readAsDataURL(event.data);
                    }}
                }};
                
                updateState('listening', 'READY');
                connectionInfo.textContent = 'Connected - Click to talk';
                
            }} catch (error) {{
                console.error("Microphone error:", error);
                updateState('error', 'MIC ACCESS DENIED');
                connectionInfo.textContent = 'Microphone access required';
            }}
        }}

        function toggleRecording() {{
            if (!mediaRecorder) {{
                console.log("MediaRecorder not initialized");
                return;
            }}
            
            if (isAISpeaking) {{
                console.log("AI is speaking, ignoring click");
                return;
            }}
            
            if (isRecording) {{
                mediaRecorder.stop();
                updateState('processing', 'PROCESSING...');
            }} else {{
                mediaRecorder.start();
                updateState('user-speaking', 'LISTENING...');
            }}
        }}

        function connect() {{
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${{protocol}}//${{window.location.host}}/ws`;
            
            console.log("Connecting to:", wsUrl);
            ws = new WebSocket(wsUrl);

            ws.onopen = () => {{
                console.log("WebSocket connected");
                reconnectAttempts = 0;
                initMicrophone();
            }};
            
            ws.onmessage = (event) => {{
                try {{
                    const msg = JSON.parse(event.data);
                    console.log("Received message:", msg.type);
                    
                    if (msg.type === 'audio_chunk') {{
                        audioQueue.push(msg.data);
                        if (!isPlayingFromQueue) {{
                            playAudioFromQueue();
                        }}
                    }} else if (msg.type === 'audio_stop') {{
                        const checkDone = setInterval(() => {{
                            if (audioQueue.length === 0 && !isPlayingFromQueue) {{
                                clearInterval(checkDone);
                                isAISpeaking = false;
                                updateState('listening', 'READY');
                            }}
                        }}, 200);
                    }} else if (msg.type === 'status') {{
                        const statusMap = {{
                            'listening': 'READY',
                            'thinking': 'THINKING...',
                            'processing': 'PROCESSING...'
                        }};
                        updateState(msg.message, statusMap[msg.message] || msg.message.toUpperCase());
                    }} else if (msg.type === 'error') {{
                        console.error("Server error:", msg.message);
                        updateState('error', 'ERROR OCCURRED');
                    }}
                }} catch (e) {{
                    console.error("Message parsing error:", e);
                }}
            }};
            
            ws.onclose = (event) => {{
                console.log("WebSocket closed:", event.code, event.reason);
                updateState('error', 'CONNECTION LOST');
                connectionInfo.textContent = 'Reconnecting...';
                
                // Exponential backoff reconnection
                if (reconnectAttempts < 5) {{
                    const delay = Math.pow(2, reconnectAttempts) * 1000;
                    setTimeout(() => {{
                        reconnectAttempts++;
                        connect();
                    }}, delay);
                }}
            }};
            
            ws.onerror = (error) => {{
                console.error("WebSocket error:", error);
            }};
        }}

        // Initialize on page load
        window.addEventListener('load', connect);
        
        // Click handler for recording
        document.body.addEventListener('click', toggleRecording);
        
        // Prevent accidental navigation
        window.addEventListener('beforeunload', (e) => {{
            if (ws && ws.readyState === WebSocket.OPEN) {{
                ws.close();
            }}
        }});
    </script>
</body>
</html>
"""
    return web.Response(text=html_content, content_type='text/html')

# --- APPLICATION SETUP & LAUNCH ---
def create_app():
    app = web.Application()
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True, 
            expose_headers="*", 
            allow_headers="*", 
            allow_methods="*"
        )
    })
    
    app.router.add_get('/', serve_cosmic_interface)
    app.router.add_get('/ws', websocket_handler)
    
    # Add CORS to all routes
    for route in list(app.router.routes()):
        cors.add(route)
    
    return app

if __name__ == "__main__":
    # Check for ffmpeg availability
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        logger.info("✅ ffmpeg is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("❌ ffmpeg not found. Audio transcription may not work.")
        logger.error("Install with: apt-get install ffmpeg (Ubuntu) or brew install ffmpeg (Mac)")
    
    load_conversation_history()
    app = create_app()
    logger.info(f"🚀 Launching NEXUS on http://{HOST}:{PORT}")
    web.run_app(app, host=HOST, port=PORT)
