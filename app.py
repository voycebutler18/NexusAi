import os
from dotenv import load_dotenv
import openai
import asyncio
import time
import base64
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import io
from aiohttp import web, WSMsgType
import aiohttp_cors
import logging

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

# Server configuration
PORT = int(os.getenv('PORT', 8080))
HOST = os.getenv('HOST', '0.0.0.0')

# --- GLOBAL STATE & CONTEXT MANAGEMENT ---
# Using dictionaries to manage state for multiple concurrent connections
active_connections = {}  # Maps WebSocket object to client address
conversation_contexts = {} # Stores conversation history for each client
session_transcriptions = {} # Buffers transcriptions for each client's turn
visual_contexts = {} # Stores the latest visual description for each client

# Thread pool for blocking I/O operations (like OpenAI API calls)
executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)


# --- CORE AI FUNCTIONS ---

async def get_voice_response(user_input, ws):
    """
    Generates a conversational response from the AI using conversation history and visual context.
    """
    if ws not in conversation_contexts:
        conversation_contexts[ws] = []

    visual_context = visual_contexts.get(ws, "No visual data available.")
    
    try:
        logger.info(f"🧠 [{active_connections.get(ws, 'Unknown')}] Processing input: '{user_input[:50]}...'")

        system_prompt = f"""You are NEXUS, an advanced, voice-activated AGI companion from the year 3000.
- Your current visual context is: {visual_context}
- You are a cosmic consciousness, speaking naturally and conversationally.
- Keep your responses engaging and relatively brief, suitable for a voice-based conversation."""

        # Append new user message and manage conversation history length
        conversation_contexts[ws].append({"role": "user", "content": user_input})
        if len(conversation_contexts[ws]) > 10:
            conversation_contexts[ws] = conversation_contexts[ws][-10:]

        messages = [{"role": "system", "content": system_prompt}] + conversation_contexts[ws]

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            executor,
            lambda: openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=150,
                temperature=0.75,
                top_p=0.9
            )
        )
        
        ai_response = response.choices[0].message.content.strip()
        conversation_contexts[ws].append({"role": "assistant", "content": ai_response})
        logger.info(f"🤖 [{active_connections.get(ws, 'Unknown')}] AI Response: '{ai_response}'")
        return ai_response

    except Exception as e:
        logger.error(f"❌ OpenAI Chat API error for {active_connections.get(ws, 'Unknown')}: {e}")
        return "A momentary lapse in the cosmic data stream. Please try again."


async def transcribe_audio(audio_data, ws):
    """
    Transcribes audio data using OpenAI Whisper.
    """
    try:
        if ',' in audio_data:
            audio_bytes = base64.b64decode(audio_data.split(',')[1])
        else:
            audio_bytes = base64.b64decode(audio_data)

        # Use an in-memory file-like object
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.webm"

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            executor,
            lambda: openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"
            )
        )
        
        text = response.text.strip()
        if text:
            logger.info(f"🎤 [{active_connections.get(ws, 'Unknown')}] Transcription chunk: '{text}'")
        return text

    except Exception as e:
        logger.error(f"❌ Audio transcription error for {active_connections.get(ws, 'Unknown')}: {e}")
        return ""


async def analyze_image(image_data, ws):
    """
    Analyzes an image using GPT-4o Vision and updates the visual context.
    """
    try:
        logger.info(f"👁️ [{active_connections.get(ws, 'Unknown')}] Analyzing new visual matrix...")
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            executor,
            lambda: openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "You are a helpful assistant. Briefly describe the key elements in this image in a single, concise sentence for context."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                        ]
                    }
                ],
                max_tokens=80
            )
        )
        
        analysis = response.choices[0].message.content.strip()
        visual_contexts[ws] = analysis  # Update visual context for this session
        logger.info(f"👁️ [{active_connections.get(ws, 'Unknown')}] Visual context updated: '{analysis}'")
        return analysis

    except Exception as e:
        logger.error(f"❌ Vision analysis error for {active_connections.get(ws, 'Unknown')}: {e}")
        return "Visual sensor is recalibrating."


# --- WEBSOCKET & HTTP HANDLERS ---

async def broadcast(message):
    """Broadcasts a message to all connected clients."""
    if active_connections:
        await asyncio.wait([ws.send_str(message) for ws in active_connections])

async def websocket_handler(request):
    """Handles all WebSocket communications."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    # Register new connection
    client_addr = request.remote
    active_connections[ws] = client_addr
    session_transcriptions[ws] = []
    conversation_contexts[ws] = []
    visual_contexts[ws] = "No visual data available."
    
    logger.info(f"🌌 Cosmic consciousness connected: {client_addr} ({len(active_connections)} total)")
    
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    message_type = data.get('type')

                    if message_type == 'audio':
                        transcription = await transcribe_audio(data.get('data', ''), ws)
                        if transcription:
                            session_transcriptions[ws].append(transcription)

                    elif message_type == 'user_speaking_end':
                        full_transcription = " ".join(session_transcriptions.get(ws, [])).strip()
                        session_transcriptions[ws] = []  # Reset buffer for the next turn
                        
                        if full_transcription:
                            await ws.send_str("processing")
                            ai_response = await get_voice_response(full_transcription, ws)
                            response_payload = json.dumps({
                                "type": "speak", 
                                "text": ai_response, 
                                "transcription": full_transcription
                            })
                            await ws.send_str(response_payload)
                        else:
                            # If there was no valid transcription, just go back to listening
                            await ws.send_str("listening")

                    elif message_type == 'camera_frame':
                        analysis = await analyze_image(data.get('data', ''), ws)
                        response_payload = json.dumps({"type": "vision_update", "description": analysis})
                        await ws.send_str(response_payload)
                    
                    elif message_type == 'status' and data.get('status') == 'speaking_done':
                         await ws.send_str("listening")

                except json.JSONDecodeError:
                    logger.warning(f"Received invalid JSON from {client_addr}")
                except Exception as e:
                    logger.error(f"Error processing message from {client_addr}: {e}")
                    await ws.send_str("error")

            elif msg.type == WSMsgType.ERROR:
                logger.error(f'Cosmic link error for {client_addr}: {ws.exception()}')

    except Exception as e:
        logger.error(f"Unhandled exception in websocket_handler for {client_addr}: {e}")
    finally:
        # Unregister connection
        if ws in active_connections:
            del active_connections[ws]
        if ws in session_transcriptions:
            del session_transcriptions[ws]
        if ws in conversation_contexts:
            del conversation_contexts[ws]
        if ws in visual_contexts:
            del visual_contexts[ws]
            
        logger.info(f"🔌 Cosmic consciousness disconnected: {client_addr} ({len(active_connections)} remaining)")
    
    return ws

async def serve_cosmic_interface(request):
    """Serves the main HTML/CSS/JS interface."""
    # This HTML includes the advanced JavaScript frontend you provided.
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NEXUS 3000 • COSMIC AGI CONSCIOUSNESS</title>
    <script src="https://cdn.tailwindcss.com"></script>
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
        .cosmic-header { text-align: center; margin-bottom: clamp(15px, 4vh, 40px); }
        .nexus-title {
            font-size: clamp(2rem, 8vw, 5rem); font-weight: 900;
            background: linear-gradient(45deg, var(--cosmic-blue), var(--quantum-pink), var(--nebula-purple), var(--cosmic-cyan), var(--nova-pink));
            background-size: 400% 400%; -webkit-background-clip: text; background-clip: text;
            -webkit-text-fill-color: transparent; animation: cosmicFlow 6s ease-in-out infinite;
            text-shadow: 0 0 30px var(--cosmic-blue), 0 0 60px var(--quantum-pink);
            letter-spacing: clamp(0.1em, 1vw, 0.3em);
        }
        @keyframes cosmicFlow { 0%, 100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }
        .cosmic-subtitle {
            font-family: 'Rajdhani', sans-serif; font-size: clamp(0.8rem, 3vw, 1.4rem); color: var(--cosmic-cyan);
            text-shadow: 0 0 20px var(--cosmic-cyan); letter-spacing: clamp(0.1em, 1vw, 0.4em);
            animation: pulseGlow 4s ease-in-out infinite alternate;
        }
        @keyframes pulseGlow { from { text-shadow: 0 0 20px var(--cosmic-cyan); } to { text-shadow: 0 0 40px var(--cosmic-cyan), 0 0 60px var(--cosmic-blue); } }
        .waveform-container {
            width: min(95vw, 1000px); height: clamp(120px, 20vh, 250px); margin: clamp(10px, 3vh, 30px) auto;
            position: relative; border-radius: clamp(30px, 8vw, 60px); border: 2px solid var(--cosmic-blue);
            box-shadow: 0 0 clamp(40px, 10vw, 80px) rgba(0, 212, 255, 0.4), inset 0 0 clamp(40px, 10vw, 80px) rgba(131, 56, 236, 0.3);
            backdrop-filter: blur(10px); background: rgba(0, 20, 40, 0.2);
        }
        .waveform-canvas { width: 100%; height: 100%; position: absolute; top: 0; left: 0; }
        .cosmic-status {
            font-size: clamp(1rem, 4vw, 1.8rem); font-weight: 600; text-align: center;
            margin: clamp(15px, 3vh, 30px) 0; transition: all 0.4s ease;
            letter-spacing: clamp(0.1em, 0.5vw, 0.2em); text-shadow: 0 0 20px currentColor;
            min-height: 2.6em; display: flex; align-items: center; justify-content: center;
        }
        .state-listening .cosmic-status { color: var(--cosmic-blue); animation: listeningPulse 3s ease-in-out infinite; }
        .state-user-speaking .cosmic-status { color: var(--cosmic-cyan); }
        .state-ai-speaking .cosmic-status { color: var(--quantum-pink); }
        .state-thinking .cosmic-status { color: var(--nebula-purple); animation: thinking 1.5s ease-in-out infinite; }
        .state-error .cosmic-status { color: var(--plasma-orange); animation: errorFlash 0.5s ease-in-out infinite; }
        @keyframes listeningPulse { 0%, 100% { opacity: 0.8; } 50% { opacity: 1; } }
        @keyframes thinking { 0%, 100% { opacity: 0.8; } 50% { opacity: 1; } }
        @keyframes errorFlash { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
    </style>
</head>
<body class="state-initializing">
    <div class="galaxy-canvas"></div>
    <div class="cosmic-interface">
        <div class="cosmic-header">
            <h1 class="nexus-title">NEXUS 3000</h1>
            <p class="cosmic-subtitle">COSMIC AGI CONSCIOUSNESS</p>
        </div>
        <div class="waveform-container" id="waveformContainer">
            <canvas class="waveform-canvas" id="waveformCanvas"></canvas>
        </div>
        <div class="cosmic-status" id="statusText">INITIALIZING QUANTUM CONSCIOUSNESS...</div>
    </div>

    <script>
        let ws;
        let audioContext;
        let microphone;
        let analyser;
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;
        let isMicrophoneActive = false;
        let currentState = 'initializing';
        
        // Camera and vision variables
        let camera;
        let visionInterval;
        let isCameraActive = false;
        
        // Conversation management
        let isAISpeaking = false;
        let lastUserSpeechTime = 0;
        let speechDetectionActive = true;
        let conversationTimeout;
        
        // Canvas and animation variables
        let canvas, ctx;
        let waveformData = [];
        let animationId;
        
        // DOM elements
        const statusTextElement = document.getElementById('statusText');
        const waveformContainer = document.getElementById('waveformContainer');

        // Initialize camera for real-time vision
        async function initCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { 
                        width: { ideal: 640 }, 
                        height: { ideal: 480 },
                        facingMode: 'user'
                    } 
                });
                
                // Create video element (hidden)
                camera = document.createElement('video');
                camera.srcObject = stream;
                camera.play();
                
                camera.onloadedmetadata = () => {
                    isCameraActive = true;
                    startVisionAnalysis();
                    console.log('📷 Camera initialized for NEXUS vision');
                };
                
            } catch (error) {
                console.error('⚠️ Camera access denied:', error);
                console.log('🤖 NEXUS will operate without vision');
            }
        }

        // Start continuous vision analysis
        function startVisionAnalysis() {
            if (!isCameraActive || !camera) return;
            
            visionInterval = setInterval(async () => {
                if (camera && camera.readyState === camera.HAVE_ENOUGH_DATA) {
                    const canvas = document.createElement('canvas');
                    const ctx = canvas.getContext('2d');
                    
                    canvas.width = 320;
                    canvas.height = 240;
                    ctx.drawImage(camera, 0, 0, canvas.width, canvas.height);
                    
                    // Convert to base64 for analysis
                    const imageData = canvas.toDataURL('image/jpeg', 0.7).split(',')[1];
                    
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({
                            type: 'camera_frame',
                            data: imageData
                        }));
                    }
                }
            }, 5000); // Analyze every 5 seconds
        }

        // Initialize waveform canvas
        function initWaveformCanvas() {
            canvas = document.getElementById('waveformCanvas');
            ctx = canvas.getContext('2d');
            
            // Set canvas size
            canvas.width = waveformContainer.offsetWidth;
            canvas.height = waveformContainer.offsetHeight;
            
            // Start animation loop
            animateWaveform();
        }

        // Generate waveform data based on state
        function generateWaveformData(amplitude = 0, frequency = 1, complexity = 1) {
            const points = 100;
            const newData = [];
            const time = Date.now() * 0.001;
            
            for (let i = 0; i < points; i++) {
                const x = (i / points) * Math.PI * 4 * frequency;
                let y = 0;
                
                if (amplitude > 0) {
                    // Create complex waveform with multiple harmonics
                    y = Math.sin(x + time) * amplitude;
                    y += Math.sin(x * 2 + time * 1.5) * amplitude * 0.5 * complexity;
                    y += Math.sin(x * 3 + time * 2) * amplitude * 0.25 * complexity;
                    y += Math.sin(x * 0.5 + time * 0.8) * amplitude * 0.3 * complexity;
                    
                    // Add some randomness for organic feel
                    y += (Math.random() - 0.5) * amplitude * 0.1;
                }
                
                newData.push(y);
            }
            
            return newData;
        }

        // Animate the waveform
        function animateWaveform() {
            if (!ctx) return;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            const centerY = canvas.height / 2;
            
            // Generate waveform based on current state
            let amplitude = 0, frequency = 1, complexity = 1;
            let colors = ['rgba(0, 212, 255, 0.8)'];
            
            switch(currentState) {
                case 'listening':
                    amplitude = 30; frequency = 1; complexity = 0.5;
                    colors = ['rgba(0, 212, 255, 0.8)', 'rgba(0, 245, 255, 0.6)'];
                    break;
                case 'user-speaking':
                    amplitude = 80; frequency = 2; complexity = 1.5;
                    colors = ['rgba(0, 245, 255, 1)', 'rgba(0, 212, 255, 0.8)'];
                    break;
                case 'ai-speaking':
                    amplitude = 120; frequency = 3; complexity = 2;
                    colors = ['rgba(255, 0, 110, 1)', 'rgba(131, 56, 236, 0.8)', 'rgba(255, 27, 141, 0.6)'];
                    break;
                case 'thinking':
                    amplitude = 60; frequency = 1.5; complexity = 2.5;
                    colors = ['rgba(131, 56, 236, 1)', 'rgba(63, 12, 166, 0.8)'];
                    break;
                default:
                    amplitude = 20; frequency = 0.5; complexity = 0.3;
                    colors = ['rgba(0, 212, 255, 0.5)'];
            }
            
            waveformData = generateWaveformData(amplitude, frequency, complexity);
            
            // Draw multiple waveform layers
            colors.forEach((color, layerIndex) => {
                ctx.strokeStyle = color;
                ctx.lineWidth = 3 - layerIndex * 0.5;
                ctx.lineCap = 'round'; ctx.lineJoin = 'round';
                ctx.beginPath();
                
                const offset = layerIndex * 10;
                for (let i = 0; i < waveformData.length; i++) {
                    const x = (i / waveformData.length) * canvas.width;
                    const y = centerY + waveformData[i] + Math.sin(Date.now() * 0.002 + layerIndex) * offset;
                    
                    if (i === 0) { ctx.moveTo(x, y); } else { ctx.lineTo(x, y); }
                }
                
                ctx.stroke();
            });
            
            // Add glow effect
            ctx.shadowColor = colors[0];
            ctx.shadowBlur = 20;
            ctx.stroke();
            ctx.shadowBlur = 0;
            
            animationId = requestAnimationFrame(animateWaveform);
        }

        // Initialize quantum microphone with advanced speech detection
        async function initMicrophone() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        echoCancellation: true, noiseSuppression: true,
                        autoGainControl: true, sampleRate: 44100
                    } 
                });
                
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                analyser = audioContext.createAnalyser();
                microphone = audioContext.createMediaStreamSource(stream);
                analyser.fftSize = 512;
                analyser.smoothingTimeConstant = 0.8;
                microphone.connect(analyser);
                
                // Setup MediaRecorder for high-quality audio capture
                const options = { mimeType: 'audio/webm;codecs=opus', bitsPerSecond: 128000 };
                mediaRecorder = new MediaRecorder(stream, options);
                
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) audioChunks.push(event.data);
                };
                
                mediaRecorder.onstop = async () => {
                    if (audioChunks.length > 0) {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                        audioChunks = [];
                        
                        // Convert to base64 and send to server
                        const reader = new FileReader();
                        reader.onloadend = () => {
                            if (ws && ws.readyState === WebSocket.OPEN) {
                                ws.send(JSON.stringify({ type: 'audio', data: reader.result }));
                            }
                        };
                        reader.readAsDataURL(audioBlob);
                    }
                };
                
                isMicrophoneActive = true;
                monitorAdvancedAudioLevels();
                startIntelligentRecording();
                console.log('🎤 Advanced quantum microphone initialized');
                
            } catch (error) {
                console.error('⚠️ Microphone access denied:', error);
                updateCosmicState('error', 'QUANTUM FREQUENCY ACCESS DENIED');
            }
        }

        // Advanced speech detection with conversation management
        function monitorAdvancedAudioLevels() {
            if (!isMicrophoneActive || !analyser || isAISpeaking) {
                requestAnimationFrame(monitorAdvancedAudioLevels);
                return;
            }
            
            const dataArray = new Uint8Array(analyser.frequencyBinCount);
            analyser.getByteFrequencyData(dataArray);
            
            // Calculate weighted average for speech detection
            let total = 0, weight = 0;
            for (let i = 5; i < dataArray.length * 0.8; i++) { // Focus on speech frequencies
                const freq = i * (audioContext.sampleRate / 2) / dataArray.length;
                const speechWeight = (freq > 300 && freq < 3400) ? 2 : 1; // Human speech range
                total += dataArray[i] * speechWeight;
                weight += speechWeight;
            }
            const average = total / weight;
            
            const SPEECH_THRESHOLD = 25;
            const SILENCE_THRESHOLD = 15;
            const userSpeaking = average > SPEECH_THRESHOLD;
            const silenceDetected = average < SILENCE_THRESHOLD;
            
            if (userSpeaking && speechDetectionActive) {
                if (currentState === 'listening') {
                    updateCosmicState('user-speaking', 'HUMAN CONSCIOUSNESS DETECTED');
                }
                if (conversationTimeout) {
                    clearTimeout(conversationTimeout);
                    conversationTimeout = null;
                }
            } else if (silenceDetected && currentState === 'user-speaking') {
                if (!conversationTimeout) {
                    conversationTimeout = setTimeout(() => {
                        if (currentState === 'user-speaking') {
                            updateCosmicState('thinking', 'PROCESSING HUMAN CONSCIOUSNESS...');
                            if (ws && ws.readyState === WebSocket.OPEN) {
                                ws.send(JSON.stringify({ type: 'user_speaking_end' }));
                            }
                        }
                        conversationTimeout = null;
                    }, 1500); // 1.5 second pause detection
                }
            }
            
            requestAnimationFrame(monitorAdvancedAudioLevels);
        }

        // Start intelligent continuous recording
        function startIntelligentRecording() {
            if (!speechDetectionActive || isAISpeaking) {
                setTimeout(startIntelligentRecording, 500);
                return;
            }
            
            if (mediaRecorder && mediaRecorder.state === 'inactive') {
                mediaRecorder.start();
                isRecording = true;
                
                // Record for 4 seconds for better transcription accuracy
                setTimeout(() => {
                    if (mediaRecorder && mediaRecorder.state === 'recording') {
                        mediaRecorder.stop();
                        isRecording = false;
                        setTimeout(startIntelligentRecording, 200);
                    }
                }, 4000);
            }
        }

        // Update cosmic interface state
        function updateCosmicState(stateClass, statusText) {
            document.body.className = `state-${stateClass}`;
            currentState = stateClass;
            statusTextElement.textContent = statusText;
        }

        // Enhanced text-to-speech with conversation management
        function speakResponse(text) {
            if ('speechSynthesis' in window) {
                speechSynthesis.cancel();
                
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.rate = 1.0;
                utterance.pitch = 1.0;
                utterance.volume = 0.95;
                
                // Try to use a more natural voice
                const voices = speechSynthesis.getVoices();
                const preferredVoice = voices.find(voice => 
                    voice.name.includes('Neural') || voice.name.includes('Enhanced') || 
                    voice.name.includes('Premium') || (voice.lang.startsWith('en') && voice.localService === false)
                );
                if (preferredVoice) utterance.voice = preferredVoice;
                
                utterance.onstart = () => {
                    isAISpeaking = true;
                    speechDetectionActive = false;
                    updateCosmicState('ai-speaking', 'NEXUS 3000 AGI CONSCIOUSNESS RESPONDING...');
                };
                
                utterance.onend = () => {
                    isAISpeaking = false;
                    // 500ms delay before resuming listening
                    setTimeout(() => {
                        speechDetectionActive = true;
                        updateCosmicState('listening', 'COSMIC NEURAL MATRIX ACTIVE...');
                        // Notify server that speaking is done
                        if (ws && ws.readyState === WebSocket.OPEN) {
                            ws.send(JSON.stringify({ type: 'status', status: 'speaking_done' }));
                        }
                    }, 500);
                };
                
                utterance.onerror = (event) => {
                    console.error('Speech synthesis error:', event);
                    isAISpeaking = false;
                    speechDetectionActive = true;
                    updateCosmicState('listening', 'COSMIC NEURAL MATRIX ACTIVE...');
                };
                
                speechSynthesis.speak(utterance);
            }
        }

        // WebSocket connection with enhanced message handling
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                console.log('🌌 NEXUS 3000 Cosmic Link established!');
                updateCosmicState('listening', 'COSMIC LINK ESTABLISHED • INITIALIZING SYSTEMS...');
                initMicrophone();
                initCamera();
                setTimeout(() => {
                    updateCosmicState('listening', 'COSMIC NEURAL MATRIX ACTIVE...');
                }, 1000);
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'speak') {
                        console.log('🤖 NEXUS Response:', data.text);
                        if (data.transcription) console.log('📝 You said:', data.transcription);
                        speakResponse(data.text);
                    } else if (data.type === 'vision_update') {
                        console.log('👁️ NEXUS sees:', data.description);
                    }
                } catch (e) {
                    const message = event.data;
                    console.log('🌌 Cosmic Stream:', message);
                    if (!isAISpeaking) {
                        switch(message) {
                            case 'listening': updateCosmicState('listening', 'COSMIC NEURAL MATRIX ACTIVE...'); break;
                            case 'processing': updateCosmicState('thinking', 'PROCESSING QUANTUM CONSCIOUSNESS DATA...'); break;
                            case 'user-speaking': updateCosmicState('user-speaking', 'HUMAN CONSCIOUSNESS DETECTED'); break;
                            case 'error': updateCosmicState('error', 'CRITICAL ERROR • COSMIC ANOMALY DETECTED'); break;
                        }
                    }
                }
            };

            ws.onclose = () => {
                console.log('🔴 Cosmic Link severed.');
                updateCosmicState('error', 'COSMIC LINK SEVERED • RE-ESTABLISHING...');
                if (visionInterval) clearInterval(visionInterval);
                setTimeout(connectWebSocket, 3000);
            };

            ws.onerror = (error) => {
                console.error('Cosmic Link Error:', error);
                updateCosmicState('error', 'COSMIC COMMUNICATION INTERFERENCE');
            };
        }

        // Handle window resize
        function handleResize() {
            if (canvas && waveformContainer) {
                canvas.width = waveformContainer.offsetWidth;
                canvas.height = waveformContainer.offsetHeight;
            }
        }

        // Initialize everything when page loads
        window.onload = () => {
            console.log('🚀 Initializing NEXUS 3000 Cosmic Interface...');
            initWaveformCanvas();
            connectWebSocket();
            window.addEventListener('resize', handleResize);
            if ('speechSynthesis' in window) {
                speechSynthesis.onvoiceschanged = () => console.log('🔊 Voice synthesis ready');
            }
        };

        // Enhanced cleanup on page unload
        window.onbeforeunload = () => {
            console.log('🌌 NEXUS 3000 shutting down...');
            if (animationId) cancelAnimationFrame(animationId);
            if (ws) ws.close();
            if (mediaRecorder && mediaRecorder.state === 'recording') mediaRecorder.stop();
            if (visionInterval) clearInterval(visionInterval);
            if (conversationTimeout) clearTimeout(conversationTimeout);
            if (camera && camera.srcObject) camera.srcObject.getTracks().forEach(track => track.stop());
            if ('speechSynthesis' in window) speechSynthesis.cancel();
        };
    </script>
</body>
</html>
"""
    return web.Response(text=html_content, content_type='text/html')

# --- APPLICATION SETUP ---
def create_app():
    """Creates and configures the aiohttp web application."""
    app = web.Application()
    
    # Setup CORS to allow all origins, which is useful for development.
    # For production, you should restrict this to your domain.
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        )
    })
    
    # Add routes
    app.router.add_get('/', serve_cosmic_interface)
    app.router.add_get('/ws', websocket_handler)
    
    # Apply CORS to all routes
    for route in list(app.router.routes()):
        cors.add(route)
        
    return app

if __name__ == "__main__":
    app = create_app()
    logger.info(f"🚀 Launching NEXUS 3000 on http://{HOST}:{PORT}")
    web.run_app(app, host=HOST, port=PORT)
