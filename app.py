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

# Load environment variables from a .env file
load_dotenv()

# --- CONFIGURATION AND LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Securely load OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    logger.error("❌ OpenAI API key not found in environment variables (OPENAI_API_KEY).")
    raise ValueError("OPENAI_API_KEY is required.")

logger.info("✅ OpenAI API key loaded successfully.")

# Server and persistence configuration
PORT = int(os.getenv('PORT', 8080))
HOST = os.getenv('HOST', '0.0.0.0')
HISTORY_FILE = "conversation_history.json"

# --- GLOBAL STATE & CONTEXT MANAGEMENT ---
active_connections = {}
conversation_context = [] # Centralized context for our single user
session_transcriptions = {}
visual_context = "No visual data available."

executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

# --- CONVERSATION PERSISTENCE ---

def load_conversation_history():
    """Loads conversation history from the JSON file at startup."""
    global conversation_context
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                conversation_context = json.load(f)
            logger.info(f"🧠 Conversation history loaded from {HISTORY_FILE}.")
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"⚠️ Could not load conversation history: {e}. Starting fresh.")
        conversation_context = []

def save_conversation_history():
    """Saves the current conversation history to the JSON file."""
    global conversation_context
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(conversation_context, f, indent=2)
    except IOError as e:
        logger.error(f"⚠️ Could not save conversation history: {e}.")


# --- CORE AI FUNCTIONS ---

async def get_voice_response(user_input, is_greeting=False):
    """
    Generates a conversational response from the AI.
    """
    global conversation_context, visual_context
    
    try:
        if is_greeting:
            logger.info("👋 Generating a dynamic greeting...")
            system_prompt = "You are NEXUS, an advanced AGI. Start our conversation with a creative, short, and welcoming greeting."
            messages = [{"role": "system", "content": system_prompt}]
        else:
            logger.info(f"🧠 Processing input: '{user_input[:50]}...'")
            system_prompt = f"""You are NEXUS, an advanced, voice-activated AGI companion from the year 3000.
- Your current visual context is: {visual_context}
- Respond naturally and conversationally. Keep responses engaging and relatively brief for voice chat."""
            conversation_context.append({"role": "user", "content": user_input})
            # Keep the context from growing too large
            if len(conversation_context) > 12:
                conversation_context = conversation_context[-12:]
            messages = [{"role": "system", "content": system_prompt}] + conversation_context

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            executor,
            lambda: openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=150,
                temperature=0.8,
                top_p=1.0
            )
        )
        
        ai_response = response.choices[0].message.content.strip()
        
        if not is_greeting:
            conversation_context.append({"role": "assistant", "content": ai_response})
            save_conversation_history() # Save after every successful turn
        
        logger.info(f"🤖 AI Response: '{ai_response}'")
        return ai_response

    except Exception as e:
        logger.error(f"❌ OpenAI Chat API error: {e}")
        return "A momentary lapse in the cosmic data stream. Please try again."

async def transcribe_audio(audio_data):
    """Transcribes audio data using OpenAI Whisper."""
    try:
        audio_bytes = base64.b64decode(audio_data.split(',')[1])
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.webm"

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            executor,
            lambda: openai.audio.transcriptions.create(model="whisper-1", file=audio_file)
        )
        text = response.text.strip()
        if text:
            logger.info(f"🎤 Transcription chunk: '{text}'")
        return text

    except Exception as e:
        logger.error(f"❌ Audio transcription error: {e}")
        return ""

async def analyze_image(image_data):
    """Analyzes an image and updates the visual context."""
    global visual_context
    try:
        logger.info("👁️ Analyzing new visual matrix...")
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            executor,
            lambda: openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": "Briefly describe the key elements in this image in one concise sentence for context."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                ]}],
                max_tokens=80
            )
        )
        analysis = response.choices[0].message.content.strip()
        visual_context = analysis
        logger.info(f"👁️ Visual context updated: '{analysis}'")
        return analysis
    except Exception as e:
        logger.error(f"❌ Vision analysis error: {e}")
        return "Visual sensor is recalibrating."

# --- WEBSOCKET & HTTP HANDLERS ---

async def websocket_handler(request):
    """Handles all WebSocket communications."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    client_addr = request.remote
    active_connections[ws] = client_addr
    session_transcriptions[ws] = []
    
    logger.info(f"🌌 Cosmic consciousness connected: {client_addr} ({len(active_connections)} total)")
    
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    msg_type = data.get('type')

                    if msg_type == 'audio':
                        transcription = await transcribe_audio(data.get('data', ''))
                        if transcription:
                            session_transcriptions[ws].append(transcription)

                    elif msg_type == 'user_speaking_end':
                        full_transcription = " ".join(session_transcriptions.get(ws, [])).strip()
                        session_transcriptions[ws] = []
                        
                        if full_transcription:
                            await ws.send_str("processing")
                            ai_response = await get_voice_response(full_transcription)
                            await ws.send_str(json.dumps({"type": "speak", "text": ai_response, "transcription": full_transcription}))
                        else:
                            await ws.send_str("listening")

                    elif msg_type == 'camera_frame':
                        analysis = await analyze_image(data.get('data', ''))
                        await ws.send_str(json.dumps({"type": "vision_update", "description": analysis}))
                    
                    elif msg_type == 'status':
                        status = data.get('status')
                        if status == 'speaking_done':
                             await ws.send_str("listening")
                        # Client is ready, generate and send a greeting from the AI
                        elif status == 'ready_for_conversation':
                            greeting = await get_voice_response(None, is_greeting=True)
                            await ws.send_str(json.dumps({"type": "speak", "text": greeting}))

                except Exception as e:
                    logger.error(f"Error processing message from {client_addr}: {e}")
                    await ws.send_str("error")

    finally:
        del active_connections[ws]
        del session_transcriptions[ws]
        logger.info(f"🔌 Cosmic consciousness disconnected: {client_addr} ({len(active_connections)} remaining)")
    
    return ws

async def serve_cosmic_interface(request):
    """Serves the main HTML/CSS/JS interface with updated pause timing."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NEXUS 3000 • COSMIC AGI CONSCIOUSNESS</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --cosmic-blue: #00D4FF; --quantum-pink: #FF006E; --nebula-purple: #8338EC;
            --stellar-white: #FFFFFF; --void-black: #000000; --deep-space: #0A0A0F;
            --cosmic-cyan: #00F5FF; --nova-pink: #FF1B8D; --plasma-orange: #FF6B35;
        }
        body {
            font-family: 'Orbitron', monospace; background: var(--void-black); color: var(--stellar-white);
            overflow: hidden; height: 100vh; position: relative;
        }
        .galaxy-canvas {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: 1;
            background: radial-gradient(ellipse at 15% 25%, #1a0033 0%, transparent 25%),
                        radial-gradient(ellipse at 85% 75%, #0d1421 0%, transparent 25%),
                        radial-gradient(ellipse 800px 300px at 30% 40%, rgba(138, 43, 226, 0.4) 0%, transparent 40%),
                        radial-gradient(ellipse 600px 200px at 70% 60%, rgba(255, 0, 110, 0.3) 0%, transparent 35%),
                        linear-gradient(135deg, #000011 0%, #0a0a0f 50%, #000033 100%);
            animation: galaxyRotation 120s linear infinite;
        }
        @keyframes galaxyRotation { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        .cosmic-interface {
            position: relative; z-index: 10; display: flex; flex-direction: column; align-items: center;
            justify-content: center; min-height: 100vh; padding: clamp(10px, 3vw, 30px);
        }
        .nexus-title {
            font-size: clamp(2rem, 8vw, 5rem); font-weight: 900;
            background: linear-gradient(45deg, var(--cosmic-blue), var(--quantum-pink), var(--nebula-purple));
            background-size: 200% 200%; -webkit-background-clip: text; background-clip: text;
            -webkit-text-fill-color: transparent; animation: cosmicFlow 6s ease-in-out infinite;
            text-shadow: 0 0 30px var(--cosmic-blue); letter-spacing: clamp(0.1em, 1vw, 0.3em);
            text-align: center;
        }
        @keyframes cosmicFlow { 0%, 100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }
        .waveform-container {
            width: min(95vw, 800px); height: clamp(120px, 20vh, 200px); margin: clamp(10px, 3vh, 30px) auto;
            position: relative; border-radius: 20px; border: 2px solid var(--cosmic-blue);
            box-shadow: 0 0 30px rgba(0, 212, 255, 0.4), inset 0 0 30px rgba(131, 56, 236, 0.3);
            backdrop-filter: blur(5px); background: rgba(0, 20, 40, 0.2);
        }
        .waveform-canvas { width: 100%; height: 100%; position: absolute; top: 0; left: 0; }
        .cosmic-status {
            font-family: 'Rajdhani', sans-serif; font-size: clamp(1.1rem, 4vw, 1.5rem); font-weight: 600; text-align: center;
            margin: clamp(15px, 3vh, 30px) 0; transition: all 0.4s ease;
            text-shadow: 0 0 15px currentColor; min-height: 2.2em;
        }
        .state-listening .cosmic-status { color: var(--cosmic-blue); animation: pulse 3s ease-in-out infinite; }
        .state-user-speaking .cosmic-status { color: var(--cosmic-cyan); }
        .state-ai-speaking .cosmic-status { color: var(--quantum-pink); }
        .state-thinking .cosmic-status { color: var(--nebula-purple); }
        .state-error .cosmic-status { color: var(--plasma-orange); }
        @keyframes pulse { 0%, 100% { opacity: 0.7; } 50% { opacity: 1; } }
    </style>
</head>
<body class="state-initializing">
    <div class="galaxy-canvas"></div>
    <div class="cosmic-interface">
        <h1 class="nexus-title">NEXUS 3000</h1>
        <div class="waveform-container" id="waveformContainer">
            <canvas class="waveform-canvas" id="waveformCanvas"></canvas>
        </div>
        <div class="cosmic-status" id="statusText">INITIALIZING QUANTUM CONSCIOUSNESS...</div>
    </div>
    <script>
        let ws, audioContext, microphone, analyser, mediaRecorder, camera, visionInterval, canvas, ctx, animationId;
        let audioChunks = [], waveformData = [];
        let isMicrophoneActive = false, isCameraActive = false;
        let isAISpeaking = false, speechDetectionActive = true, conversationTimeout = null;
        let currentState = 'initializing';
        
        const statusTextElement = document.getElementById('statusText');
        const waveformContainer = document.getElementById('waveformContainer');

        function updateCosmicState(stateClass, statusText) {
            document.body.className = `state-${stateClass}`;
            currentState = stateClass;
            statusTextElement.textContent = statusText;
        }
        
        async function initCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 } } });
                camera = document.createElement('video');
                camera.srcObject = stream;
                camera.play();
                camera.onloadedmetadata = () => {
                    isCameraActive = true;
                    visionInterval = setInterval(() => {
                        if (ws && ws.readyState === WebSocket.OPEN && camera.readyState === camera.HAVE_ENOUGH_DATA) {
                            const canvas = document.createElement('canvas');
                            canvas.width = 320; canvas.height = 240;
                            canvas.getContext('2d').drawImage(camera, 0, 0, canvas.width, canvas.height);
                            const imageData = canvas.toDataURL('image/jpeg', 0.7).split(',')[1];
                            ws.send(JSON.stringify({ type: 'camera_frame', data: imageData }));
                        }
                    }, 5000);
                };
            } catch (error) { console.error('⚠️ Camera access denied:', error); }
        }

        function animateWaveform() {
            if (!ctx) return;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const centerY = canvas.height / 2;
            let amp = 0, freq = 1, comp = 1, colors = ['#00D4FF'];
            switch(currentState) {
                case 'listening': amp = 25; freq = 1; comp = 0.5; colors = ['#00D4FF', '#00F5FF']; break;
                case 'user-speaking': amp = 60; freq = 2; comp = 1.5; colors = ['#00F5FF', '#00D4FF']; break;
                case 'ai-speaking': amp = 90; freq = 3; comp = 2; colors = ['#FF006E', '#8338EC']; break;
                case 'thinking': amp = 40; freq = 1.5; comp = 2.5; colors = ['#8338EC', '#3F0CA6']; break;
                default: amp = 15;
            }
            const time = Date.now() * 0.001;
            waveformData = Array.from({length: 100}, (_, i) => {
                const x = (i / 100) * Math.PI * 4 * freq;
                let y = Math.sin(x + time) * amp;
                y += Math.sin(x * 2 + time * 1.5) * amp * 0.5 * comp;
                return y + (Math.random() - 0.5) * amp * 0.1;
            });
            colors.forEach((color, layerIndex) => {
                ctx.strokeStyle = color; ctx.lineWidth = 2.5 - layerIndex;
                ctx.beginPath();
                for (let i = 0; i < waveformData.length; i++) {
                    const x = (i / waveformData.length) * canvas.width;
                    const y = centerY + waveformData[i] + Math.sin(time * 2 + layerIndex) * (layerIndex * 10);
                    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
                }
                ctx.stroke();
            });
            animationId = requestAnimationFrame(animateWaveform);
        }

        async function initMicrophone() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                analyser = audioContext.createAnalyser();
                microphone = audioContext.createMediaStreamSource(stream);
                microphone.connect(analyser);
                mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
                mediaRecorder.ondataavailable = e => e.data.size > 0 && audioChunks.push(e.data);
                mediaRecorder.onstop = () => {
                    if (audioChunks.length) {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                        audioChunks = [];
                        const reader = new FileReader();
                        reader.onloadend = () => ws.send(JSON.stringify({ type: 'audio', data: reader.result }));
                        reader.readAsDataURL(audioBlob);
                    }
                };
                isMicrophoneActive = true;
                console.log("🎤 Microphone Initialized");
                monitorAdvancedAudioLevels();
                startIntelligentRecording();
            } catch (error) {
                console.error('⚠️ Microphone access denied:', error);
                updateCosmicState('error', 'MICROPHONE ACCESS DENIED');
            }
        }

        function monitorAdvancedAudioLevels() {
            if (!isMicrophoneActive || !analyser || isAISpeaking) return requestAnimationFrame(monitorAdvancedAudioLevels);
            const dataArray = new Uint8Array(analyser.frequencyBinCount);
            analyser.getByteFrequencyData(dataArray);
            const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
            const userSpeaking = average > 15;
            const silenceDetected = average < 10;
            
            if (userSpeaking && speechDetectionActive) {
                if (currentState === 'listening') updateCosmicState('user-speaking', 'LISTENING...');
                if (conversationTimeout) clearTimeout(conversationTimeout);
                conversationTimeout = null;
            } else if (silenceDetected && currentState === 'user-speaking') {
                if (!conversationTimeout) {
                    conversationTimeout = setTimeout(() => {
                        updateCosmicState('thinking', 'PROCESSING...');
                        ws.send(JSON.stringify({ type: 'user_speaking_end' }));
                        conversationTimeout = null;
                    }, 2000); // 2-second pause detection
                }
            }
            requestAnimationFrame(monitorAdvancedAudioLevels);
        }

        function startIntelligentRecording() {
            if (!speechDetectionActive || isAISpeaking) return setTimeout(startIntelligentRecording, 500);
            if (mediaRecorder && mediaRecorder.state === 'inactive') {
                mediaRecorder.start();
                setTimeout(() => {
                    if (mediaRecorder.state === 'recording') mediaRecorder.stop();
                    setTimeout(startIntelligentRecording, 100);
                }, 4000);
            }
        }

        function speakResponse(text) {
            if (!('speechSynthesis' in window)) return;
            speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.voice = speechSynthesis.getVoices().find(v => v.lang.startsWith('en') && v.name.includes('Neural')) || null;
            utterance.rate = 1.05;
            utterance.pitch = 1.0;
            utterance.onstart = () => {
                isAISpeaking = true; speechDetectionActive = false;
                updateCosmicState('ai-speaking', 'RESPONDING...');
            };
            utterance.onend = () => {
                isAISpeaking = false;
                setTimeout(() => {
                    speechDetectionActive = true;
                    updateCosmicState('listening', 'AWAITING INPUT...');
                    ws.send(JSON.stringify({ type: 'status', status: 'speaking_done' }));
                }, 500); // 0.5 second delay
            };
            speechSynthesis.speak(utterance);
        }

        // NEW, more robust function to initialize the connection
        function initializeSystems() {
            console.log("🚀 Initializing NEXUS 3000 Cosmic Interface...");
            canvas = document.getElementById('waveformCanvas');
            ctx = canvas.getContext('2d');
            canvas.width = waveformContainer.offsetWidth;
            canvas.height = waveformContainer.offsetHeight;
            animateWaveform();
            
            // This function handles the logic of waiting for voices to be ready
            function connectWhenReady() {
                // Tell the server we are ready for the initial greeting
                console.log("🔊 Voice engine ready. Signaling server.");
                ws.send(JSON.stringify({ type: 'status', status: 'ready_for_conversation' }));
                initMicrophone();
                initCamera();
            }

            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

            ws.onopen = () => {
                console.log('🌌 NEXUS 3000 Cosmic Link established!');
                updateCosmicState('listening', 'ESTABLISHING LINK...');
                
                // Wait for the browser's voice synthesis engine to be ready
                if (speechSynthesis.getVoices().length === 0) {
                    speechSynthesis.onvoiceschanged = connectWhenReady;
                } else {
                    connectWhenReady();
                }
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'speak') speakResponse(data.text);
                    else if (data.type === 'vision_update') console.log('👁️ NEXUS sees:', data.description);
                } catch (e) {
                    const state = event.data;
                    if (!isAISpeaking) {
                        if (state === 'listening') updateCosmicState('listening', 'AWAITING INPUT...');
                        else if (state === 'processing') updateCosmicState('thinking', 'PROCESSING...');
                        else if (state === 'error') updateCosmicState('error', 'SYSTEM ANOMALY DETECTED');
                    }
                }
            };
            ws.onclose = () => {
                updateCosmicState('error', 'LINK SEVERED. RECONNECTING...');
                if (visionInterval) clearInterval(visionInterval);
                setTimeout(initializeSystems, 3000);
            };
        }

        window.onload = initializeSystems;
    </script>
</body>
</html>
"""
    return web.Response(text=html_content, content_type='text/html')

# --- APPLICATION SETUP & LAUNCH ---
def create_app():
    """Creates and configures the aiohttp web application."""
    app = web.Application()
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(allow_credentials=True, expose_headers="*", allow_headers="*", allow_methods="*")
    })
    app.router.add_get('/', serve_cosmic_interface)
    app.router.add_get('/ws', websocket_handler)
    for route in list(app.router.routes()):
        cors.add(route)
    return app

if __name__ == "__main__":
    load_conversation_history() # Load memory on start
    app = create_app()
    logger.info(f"🚀 Launching NEXUS 3000 on http://{HOST}:{PORT}")
    web.run_app(app, host=HOST, port=PORT)
