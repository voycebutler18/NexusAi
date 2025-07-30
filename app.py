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
    # Emotion detector is kept as a free, local feature
    emotion_detector = FER(mtcnn=True)
    logger.info("✅ OpenAI client and emotion detector initialized.")
except Exception as e:
    logger.error(f"❌ Failed to initialize clients: {e}")
    raise

# --- GLOBAL STATE & CONTEXT MANAGEMENT ---
active_connections = {}
conversation_context = []
session_transcriptions = {}
visual_context = {"description": "No visual data available.", "emotion": "unknown"}

executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

# --- CONVERSATION PERSISTENCE ---
def load_conversation_history():
    global conversation_context
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                conversation_context = json.load(f)
            logger.info(f"🧠 Conversation history loaded from {HISTORY_FILE}.")
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

async def get_voice_response(user_input):
    """Generates a text response from GPT-4o based on conversation and sensory input."""
    global conversation_context, visual_context
    
    logger.info(f"🧠 Processing input: '{user_input[:50]}...'")
    
    full_context = f"""
    - Visuals: {visual_context['description']}
    - Detected Emotion: {visual_context['emotion']}
    """
    
    system_prompt = f"""You are Nexus, an emotionally intelligent AI companion with a friendly, casual personality. 
You can see and detect emotions. Use this awareness to enrich the conversation.
Your current sensory context is: {full_context}
Based on this, respond naturally to the user's message. Refer to what you see or the user's emotion when it feels natural.
"""
    
    conversation_context.append({"role": "user", "content": user_input})
    if len(conversation_context) > 12:
        conversation_context = conversation_context[-12:]
        
    messages = [{"role": "system", "content": system_prompt}] + conversation_context
    
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            executor,
            lambda: openai_client.chat.completions.create(model="gpt-4o", messages=messages, max_tokens=150)
        )
        ai_response_text = response.choices[0].message.content.strip()
        conversation_context.append({"role": "assistant", "content": ai_response_text})
        save_conversation_history()
        logger.info(f"🤖 AI Text Response: '{ai_response_text}'")
        return ai_response_text
    except Exception as e:
        logger.error(f"❌ OpenAI Chat API error: {e}")
        return "I'm having a little trouble connecting to my thoughts right now."

async def analyze_image_with_emotion(image_data):
    """Analyzes an image for a general description and detects facial emotions."""
    global visual_context
    loop = asyncio.get_event_loop()

    try:
        img_bytes = base64.b64decode(image_data)
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        emotion_results = await loop.run_in_executor(executor, emotion_detector.detect_emotions, img)
        
        if emotion_results:
            top_emotion = max(emotion_results[0]['emotions'], key=emotion_results[0]['emotions'].get)
            visual_context['emotion'] = top_emotion
            logger.info(f"😊 Emotion detected: {top_emotion}")
        else:
            visual_context['emotion'] = "unknown"

        response = await loop.run_in_executor(
            executor,
            lambda: openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": "Briefly describe this scene in a casual tone."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                ]}],
                max_tokens=80
            )
        )
        description = response.choices[0].message.content.strip()
        visual_context['description'] = description
        logger.info(f"👁️ Visual description: '{description}'")

        return visual_context
    except Exception as e:
        logger.error(f"❌ Vision/Emotion analysis error: {e}")
        return {"description": "My visual sensors are offline.", "emotion": "unknown"}

# --- WEBSOCKET & HTTP HANDLERS ---
async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    client_addr = request.remote
    active_connections[ws] = client_addr
    session_transcriptions[ws] = []
    logger.info(f"🌌 Connection established: {client_addr}")
    
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                msg_type = data.get('type')

                if msg_type == 'audio_input':
                    transcription = await transcribe_audio(data.get('data', ''))
                    if transcription:
                        session_transcriptions.setdefault(ws, []).append(transcription)

                elif msg_type == 'user_speaking_end':
                    full_transcription = " ".join(session_transcriptions.get(ws, [])).strip()
                    session_transcriptions[ws] = []
                    
                    if full_transcription:
                        await ws.send_str(json.dumps({"type": "status", "message": "thinking"}))
                        ai_text = await get_voice_response(full_transcription)
                        # Reverted to send text for the browser to speak
                        await ws.send_str(json.dumps({"type": "speak", "text": ai_text}))
                    else:
                        await ws.send_str(json.dumps({"type": "status", "message": "listening"}))

                elif msg_type == 'camera_frame':
                    context = await analyze_image_with_emotion(data.get('data', ''))
                    await ws.send_str(json.dumps({"type": "vision_update", "context": context}))

    finally:
        del active_connections[ws]
        logger.info(f"🔌 Connection closed: {client_addr}")
    
    return ws

async def serve_cosmic_interface(request):
    """Serves the main HTML/CSS/JS interface."""
    # The JavaScript here is updated with the smart voice selection logic
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NEXUS • AI Companion</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Rajdhani:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root { --blue: #00D4FF; --pink: #FF006E; --purple: #8338EC; --white: #FFFFFF; --black: #000000; --cyan: #00F5FF; }
        body { font-family: 'Orbitron', monospace; background: var(--black); color: var(--white); overflow: hidden; height: 100vh; }
        .bg-canvas { position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: 1; background: radial-gradient(ellipse at 50% 50%, #0a0a23 0%, #000 70%); }
        .interface { position: relative; z-index: 10; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; padding: 20px; }
        .title { font-size: clamp(2rem, 8vw, 4rem); font-weight: 700; background: linear-gradient(45deg, var(--blue), var(--pink)); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 0 20px var(--blue); }
        .waveform-container { width: min(90vw, 700px); height: clamp(100px, 20vh, 180px); margin: 20px auto; position: relative; border: 2px solid var(--blue); border-radius: 15px; box-shadow: 0 0 20px #00d4ff4d; backdrop-filter: blur(5px); background: #00000033; }
        .waveform-canvas { width: 100%; height: 100%; }
        .status { font-family: 'Rajdhani', sans-serif; font-size: clamp(1rem, 3vw, 1.4rem); font-weight: 600; text-align: center; margin-top: 15px; transition: color 0.4s; }
        .state-listening .status { color: var(--blue); } .state-user-speaking .status { color: var(--cyan); }
        .state-ai-speaking .status { color: var(--pink); } .state-thinking .status { color: var(--purple); }
    </style>
</head>
<body class="state-initializing">
    <div class="bg-canvas"></div>
    <div class="interface">
        <h1 class="title">NEXUS</h1>
        <div class="waveform-container" id="waveformContainer"><canvas class="waveform-canvas" id="waveformCanvas"></canvas></div>
        <div class="status" id="statusText">INITIALIZING...</div>
    </div>
    <script>
        let ws, audioContext, mediaRecorder, analyser, camera, visionInterval, canvas, ctx, animationId;
        let audioChunks = [];
        let isAISpeaking = false, speechDetectionActive = true, conversationTimeout = null;
        let currentState = 'initializing';
        let preferredVoice = null; // To store the best found voice

        const statusText = document.getElementById('statusText');
        const waveformContainer = document.getElementById('waveformContainer');

        function updateState(newState, text) {
            document.body.className = `state-${newState}`;
            currentState = newState;
            statusText.textContent = text;
        }
        
        // --- SMART VOICE SELECTION (NEW) ---
        function findBestVoice() {
            return new Promise(resolve => {
                let voices = speechSynthesis.getVoices();
                if (voices.length) {
                    resolve(voices);
                    return;
                }
                speechSynthesis.onvoiceschanged = () => {
                    voices = speechSynthesis.getVoices();
                    resolve(voices);
                };
            });
        }

        async function setupVoice() {
            const voices = await findBestVoice();
            const qualityTiers = [
                { keyword: 'Neural', priority: 1 },
                { keyword: 'Microsoft', priority: 2 }, // Edge often has great voices
                { keyword: 'Google', priority: 3 },
                { keyword: 'Enhanced', priority: 4 }
            ];

            for (const tier of qualityTiers) {
                const foundVoice = voices.find(v => v.lang.startsWith('en') && v.name.includes(tier.keyword));
                if (foundVoice) {
                    preferredVoice = foundVoice;
                    console.log(`🎤 High-quality voice found: ${preferredVoice.name}`);
                    return;
                }
            }
            
            // Fallback to the first available US English voice
            preferredVoice = voices.find(v => v.lang === 'en-US');
            if(preferredVoice) console.log(`🎤 Using standard voice: ${preferredVoice.name}`);
        }

        function speakResponse(text) {
            if (!('speechSynthesis' in window)) return;
            speechSynthesis.cancel();
            
            const utterance = new SpeechSynthesisUtterance(text);
            if (preferredVoice) {
                utterance.voice = preferredVoice;
            }
            utterance.rate = 1.0;
            utterance.pitch = 1.0;

            utterance.onstart = () => {
                isAISpeaking = true; speechDetectionActive = false;
                updateState('ai-speaking', 'SPEAKING...');
            };

            utterance.onend = () => {
                isAISpeaking = false;
                setTimeout(() => {
                    speechDetectionActive = true;
                    updateState('listening', 'READY');
                }, 500);
            };

            speechSynthesis.speak(utterance);
        }

        // --- MICROPHONE & SPEECH DETECTION ---
        async function initMicrophone() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                analyser = audioContext.createAnalyser();
                audioContext.createMediaStreamSource(stream).connect(analyser);
                mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
                mediaRecorder.ondataavailable = e => e.data.size > 0 && audioChunks.push(e.data);
                mediaRecorder.onstop = () => {
                    if (audioChunks.length) {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                        audioChunks = [];
                        const reader = new FileReader();
                        reader.onloadend = () => ws.send(JSON.stringify({ type: 'audio_input', data: reader.result }));
                        reader.readAsDataURL(audioBlob);
                    }
                };
                console.log("🎤 Mic Initialized");
                monitorSpeech();
                recordContinuously();
                updateState('listening', 'READY');
            } catch (error) {
                console.error('🎤 Mic Error:', error);
                updateState('error', 'MICROPHONE ACCESS DENIED');
            }
        }
        
        function monitorSpeech() {
            if (!analyser || isAISpeaking) return requestAnimationFrame(monitorSpeech);
            const dataArray = new Uint8Array(analyser.frequencyBinCount);
            analyser.getByteFrequencyData(dataArray);
            const avg = dataArray.reduce((a, b) => a + b) / dataArray.length;
            
            if (avg > 15 && speechDetectionActive) {
                if (currentState === 'listening') updateState('user-speaking', 'LISTENING...');
                if (conversationTimeout) clearTimeout(conversationTimeout);
                conversationTimeout = null;
            } else if (avg < 10 && currentState === 'user-speaking') {
                if (!conversationTimeout) {
                    conversationTimeout = setTimeout(() => {
                        updateState('thinking', 'THINKING...');
                        ws.send(JSON.stringify({ type: 'user_speaking_end' }));
                        conversationTimeout = null;
                    }, 1500);
                }
            }
            requestAnimationFrame(monitorSpeech);
        }

        function recordContinuously() {
            if (!speechDetectionActive || isAISpeaking) return setTimeout(recordContinuously, 200);
            if (mediaRecorder && mediaRecorder.state === 'inactive') {
                mediaRecorder.start();
                setTimeout(() => {
                    if (mediaRecorder.state === 'recording') mediaRecorder.stop();
                    setTimeout(recordContinuously, 100);
                }, 3000);
            }
        }
        
        // --- WEBSOCKET & INITIALIZATION ---
        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

            ws.onopen = async () => {
                console.log('🌌 Connection Established');
                await setupVoice(); // Find the best voice before starting mic
                initMicrophone();
            };

            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                if (msg.type === 'speak') {
                    speakResponse(msg.text);
                } else if (msg.type === 'status') {
                    if (msg.message === 'thinking') updateState('thinking', 'THINKING...');
                    if (msg.message === 'listening') updateState('listening', 'READY');
                } else if (msg.type === 'vision_update') {
                    console.log("👁️ Vision Update:", msg.context);
                }
            };
            ws.onclose = () => {
                updateState('error', 'CONNECTION LOST. RECONNECTING...');
                setTimeout(connect, 3000);
            };
        }

        window.onload = () => {
            canvas = document.getElementById('waveformCanvas');
            ctx = canvas.getContext('2d');
            // Simplified animation for brevity
            if(ctx) {
                canvas.width = waveformContainer.offsetWidth;
                canvas.height = waveformContainer.offsetHeight;
            }
            connect();
        };
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
