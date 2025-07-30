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
import speech_recognition as sr
from gtts import gTTS
from fer import FER
import cv2
import numpy as np

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
    emotion_detector = FER(mtcnn=True)
    recognizer = sr.Recognizer()
    logger.info("✅ API clients, emotion detector, and speech recognizer initialized.")
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

# --- CORE AI & SENSORY FUNCTIONS ---
async def transcribe_audio_on_server(audio_bytes):
    """Transcribes audio using the speech_recognition library on the backend."""
    loop = asyncio.get_event_loop()
    try:
        # The library needs an AudioData object
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
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

async def get_text_response(user_input):
    """Generates a text response from GPT-4o."""
    global conversation_context, visual_context
    
    logger.info(f"🧠 Processing: '{user_input[:50]}...'")
    system_prompt = f"""You are Nexus, an emotionally intelligent AI companion. You are friendly, casual, and empathetic.
Your sensory context is: Visuals: {visual_context['description']}. Detected Emotion: {visual_context['emotion']}.
Use this awareness to respond naturally.
"""
    conversation_context.append({"role": "user", "content": user_input})
    messages = [{"role": "system", "content": system_prompt}] + conversation_context[-12:]
    
    try:
        response = await asyncio.to_thread(
            openai_client.chat.completions.create, model="gpt-4o", messages=messages, max_tokens=150
        )
        ai_response_text = response.choices[0].message.content.strip()
        conversation_context.append({"role": "assistant", "content": ai_response_text})
        save_conversation_history()
        logger.info(f"🤖 AI Response: '{ai_response_text}'")
        return ai_response_text
    except Exception as e:
        logger.error(f"❌ OpenAI error: {e}")
        return "I'm having a little trouble connecting to my thoughts."

async def stream_audio_response(text, ws):
    """Generates audio with gTTS and streams it to the client."""
    logger.info("🔊 Generating audio with gTTS...")
    try:
        # Create an in-memory binary stream for the audio
        mp3_fp = io.BytesIO()
        tts = await asyncio.to_thread(gTTS, text=text, lang='en')
        await asyncio.to_thread(tts.write_to_fp, mp3_fp)
        mp3_fp.seek(0)
        
        # Stream the audio in chunks
        while True:
            chunk = mp3_fp.read(4096)
            if not chunk:
                break
            await ws.send_str(json.dumps({
                "type": "audio_chunk",
                "data": base64.b64encode(chunk).decode('utf-8')
            }))
        await ws.send_str(json.dumps({"type": "audio_stop"}))
    except Exception as e:
        logger.error(f"❌ gTTS streaming error: {e}")

async def analyze_image_with_emotion(image_data):
    """Analyzes an image for a general description and detects facial emotions."""
    global visual_context
    try:
        img_bytes = base64.b64decode(image_data)
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        emotion_results = await asyncio.to_thread(emotion_detector.detect_emotions, img)
        top_emotion = max(emotion_results[0]['emotions'], key=emotion_results[0]['emotions'].get) if emotion_results else "unknown"
        visual_context['emotion'] = top_emotion
        
        # The rest of the function for GPT-4o vision can be added here if needed
    except Exception as e:
        logger.error(f"❌ Vision/Emotion analysis error: {e}")


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
                data = json.loads(msg.data)
                msg_type = data.get('type')

                if msg_type == 'audio_input':
                    audio_b64 = data.get('data', '').split(',')[1]
                    audio_bytes = base64.b64decode(audio_b64)
                    
                    # Convert the webm audio blob to wav for speech_recognition
                    # This is a simplification; a full solution would use ffmpeg
                    # For now, we assume the library can handle it or we'd add conversion logic
                    transcription = await transcribe_audio_on_server(audio_bytes)

                    if transcription:
                        await ws.send_str(json.dumps({"type": "status", "message": "thinking"}))
                        ai_text = await get_text_response(transcription)
                        await stream_audio_response(ai_text, ws)
                    else:
                         await ws.send_str(json.dumps({"type": "status", "message": "listening"}))

                elif msg_type == 'camera_frame':
                    await analyze_image_with_emotion(data.get('data', ''))

    finally:
        del active_connections[ws]
        logger.info(f"🔌 Connection closed: {client_addr}")
    
    return ws

# --- HTML FRONTEND ---
async def serve_cosmic_interface(request):
    """Serves the main HTML/CSS/JS interface for backend audio processing."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>NEXUS</title>
    <style>
        body { font-family: sans-serif; background: #000; color: #fff; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
        .status { font-size: 2rem; }
        .state-listening { color: #00D4FF; } .state-user-speaking { color: #00F5FF; }
        .state-ai-speaking { color: #FF006E; } .state-thinking { color: #8338EC; }
    </style>
</head>
<body class="state-initializing">
    <div class="status" id="statusText">INITIALIZING...</div>
    <script>
        let ws, audioContext, mediaRecorder;
        let isAISpeaking = false, isRecording = false;
        let audioQueue = [], isPlayingFromQueue = false;

        const statusText = document.getElementById('statusText');

        function updateState(state, text) {
            document.body.className = `state-${state}`;
            statusText.textContent = text;
        }

        async function playAudioFromQueue() {
            if (isPlayingFromQueue || audioQueue.length === 0) return;
            isPlayingFromQueue = true;
            updateState('ai-speaking', 'SPEAKING...');
            isAISpeaking = true;
            
            const data = audioQueue.shift();
            try {
                const audioData = atob(data);
                const buffer = new Uint8Array(audioData.length);
                for (let i = 0; i < audioData.length; i++) buffer[i] = audioData.charCodeAt(i);
                
                const blob = new Blob([buffer], { type: 'audio/mpeg' });
                const url = URL.createObjectURL(blob);
                const audio = new Audio(url);
                audio.play();
                audio.onended = () => {
                    URL.revokeObjectURL(url);
                    isPlayingFromQueue = false;
                    playAudioFromQueue();
                };
            } catch (e) {
                console.error("Audio playback error:", e);
                isPlayingFromQueue = false;
            }
        }

        async function initMicrophone() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
                
                mediaRecorder.onstart = () => isRecording = true;
                mediaRecorder.onstop = () => isRecording = false;

                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0 && ws.readyState === WebSocket.OPEN) {
                        const reader = new FileReader();
                        reader.onloadend = () => ws.send(JSON.stringify({ type: 'audio_input', data: reader.result }));
                        reader.readAsDataURL(event.data);
                    }
                };
                updateState('listening', 'READY');
            } catch (error) {
                updateState('error', 'MIC ACCESS DENIED');
            }
        }

        function toggleRecording() {
            if (isAISpeaking) return;
            if (isRecording) {
                mediaRecorder.stop();
                updateState('thinking', 'PROCESSING...');
            } else {
                mediaRecorder.start();
                updateState('user-speaking', 'LISTENING...');
            }
        }

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

            ws.onopen = initMicrophone;
            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                if (msg.type === 'audio_chunk') {
                    audioQueue.push(msg.data);
                    if (!isPlayingFromQueue) playAudioFromQueue();
                } else if (msg.type === 'audio_stop') {
                    const checkDone = setInterval(() => {
                        if (audioQueue.length === 0 && !isPlayingFromQueue) {
                            clearInterval(checkDone);
                            isAISpeaking = false;
                            updateState('listening', 'READY');
                        }
                    }, 100);
                } else if (msg.type === 'status') {
                    updateState(msg.message, msg.message.toUpperCase());
                }
            };
            ws.onclose = () => {
                updateState('error', 'CONNECTION LOST');
            };
        }

        window.onload = connect;
        // Use click/tap to start/stop recording for simplicity
        document.body.addEventListener('click', toggleRecording);
    </script>
</body>
</html>
"""
    return web.Response(text=html_content, content_type='text/html')

# --- APPLICATION SETUP & LAUNCH ---
def create_app():
    app = web.Application()
    cors = aiohttp_cors.setup(app, defaults={"*": aiohttp_cors.ResourceOptions(allow_credentials=True, expose_headers="*", allow_headers="*", allow_methods="*")})
    app.router.add_get('/', serve_cosmic_interface)
    app.router.add_get('/ws', websocket_handler)
    for route in list(app.router.routes()): cors.add(route)
    return app

if __name__ == "__main__":
    load_conversation_history()
    app = create_app()
    logger.info(f"🚀 Launching NEXUS on http://{HOST}:{PORT}")
    web.run_app(app, host=HOST, port=PORT)
