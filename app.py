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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- SECURE CONFIGURATION ---
openai.api_key = os.getenv('OPENAI_API_KEY')

if not openai.api_key:
    logger.error("❌ OpenAI API key not found!")
    raise ValueError("OPENAI_API_KEY environment variable is required")

logger.info("✅ OpenAI API key loaded securely")

# Get configuration from environment
PORT = int(os.getenv('PORT', 8080))
HOST = os.getenv('HOST', '0.0.0.0')

# --- GLOBAL STATE ---
active_connections = set()
conversation_context = []
is_speaking = False
should_stop_speaking = False
current_visual_context = ""
executor = ThreadPoolExecutor(max_workers=4)

# --- AI FUNCTIONS ---
async def get_voice_response(user_input):
    """Get voice response from OpenAI."""
    global conversation_context, current_visual_context
    
    try:
        logger.info(f"🧠 Processing voice input: {user_input[:50]}...")
        
        system_prompt = f"""You are NEXUS, an advanced voice-activated AGI companion from the year 3000.

Visual context: {current_visual_context if current_visual_context else "None"}

Respond naturally and conversationally as if speaking. Keep responses brief and engaging for voice conversation. You are a cosmic consciousness."""

        conversation_context.append({"role": "user", "content": user_input})
        if len(conversation_context) > 8:
            conversation_context = conversation_context[-8:]
        
        messages = [{"role": "system", "content": system_prompt}] + conversation_context
        
        response = await asyncio.get_event_loop().run_in_executor(
            executor, lambda: openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=100,
                temperature=0.7,
                top_p=0.9
            )
        )
        
        ai_response = response.choices[0].message.content.strip()
        conversation_context.append({"role": "assistant", "content": ai_response})
        
        logger.info(f"🤖 NEXUS response: {ai_response}")
        return ai_response
        
    except Exception as e:
        logger.error(f"❌ OpenAI API error: {e}")
        return "Neural processing interference detected. Recalibrating consciousness matrix."

async def transcribe_audio(audio_data):
    """Transcribe audio using OpenAI Whisper."""
    try:
        logger.info("🎤 Transcribing cosmic frequencies...")
        
        # Handle base64 audio data
        if ',' in audio_data:
            audio_bytes = base64.b64decode(audio_data.split(',')[1])
        else:
            audio_bytes = base64.b64decode(audio_data)
        
        # Create file-like object for Whisper
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.webm"
        
        response = await asyncio.get_event_loop().run_in_executor(
            executor, lambda: openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        )
        
        text = response.text.strip()
        logger.info(f"🗣️ Human consciousness detected: '{text}'")
        return text
        
    except Exception as e:
        logger.error(f"❌ Audio transcription error: {e}")
        return ""

async def analyze_image(image_data):
    """Analyze uploaded image for visual context."""
    global current_visual_context
    
    try:
        logger.info("👁️ Analyzing visual matrix...")
        
        response = await asyncio.get_event_loop().run_in_executor(
            executor, lambda: openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are NEXUS 3000. Describe what you see briefly for voice context."
                    },
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": "Analyze visual input"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                        ]
                    }
                ],
                max_tokens=80,
                temperature=0.7
            )
        )
        
        analysis = response.choices[0].message.content.strip()
        current_visual_context = analysis
        
        logger.info(f"👁️ Visual matrix updated: {analysis}")
        return analysis
        
    except Exception as e:
        logger.error(f"❌ Vision analysis error: {e}")
        return "Visual processing matrix temporarily offline."

# --- WEBSOCKET HANDLERS ---
async def broadcast_to_all(message):
    """Broadcast message to all active connections."""
    if not active_connections:
        return
    
    disconnected = set()
    for ws in active_connections.copy():
        try:
            await ws.send_str(message)
        except Exception:
            disconnected.add(ws)
    
    active_connections.difference_update(disconnected)

async def websocket_handler(request):
    """Handle WebSocket connections for the cosmic voice interface."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    active_connections.add(ws)
    client_addr = request.remote
    logger.info(f"🌌 Cosmic consciousness connected: {client_addr}")
    
    # Send initial awakening status
    await ws.send_str("awake_listening")
    
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    message_type = data.get('type')
                    
                    if message_type == 'audio':
                        # Handle voice input
                        audio_data = data.get('data', '')
                        if audio_data:
                            # Update status to processing
                            await broadcast_to_all("processing")
                            
                            # Transcribe audio
                            transcription = await transcribe_audio(audio_data)
                            if transcription and len(transcription.strip()) > 2:
                                # Get AI response
                                ai_response = await get_voice_response(transcription)
                                
                                # Send response for TTS
                                response_data = {
                                    "type": "speak",
                                    "text": ai_response,
                                    "transcription": transcription,
                                    "timestamp": time.time()
                                }
                                await ws.send_str(json.dumps(response_data))
                            else:
                                # No clear speech detected, return to listening
                                await broadcast_to_all("listening")
                    
                    elif message_type == 'image':
                        # Handle visual input
                        image_data = data.get('data', '')
                        if image_data:
                            analysis = await analyze_image(image_data)
                            # Visual context updated, inform interface
                            vision_data = {
                                "type": "vision_update",
                                "description": analysis,
                                "timestamp": time.time()
                            }
                            await ws.send_str(json.dumps(vision_data))
                    
                    elif message_type == 'status':
                        # Handle status updates from client
                        status = data.get('status', '')
                        if status == 'speaking_done':
                            await broadcast_to_all("listening")
                        elif status == 'user_speaking_detected':
                            await broadcast_to_all("user-speaking")
                    
                    elif message_type == 'ping':
                        await ws.send_str("pong")
                        
                except json.JSONDecodeError:
                    logger.warning(f"Invalid cosmic data from {client_addr}")
                except Exception as e:
                    logger.error(f"Error processing cosmic message: {e}")
                    await ws.send_str("error")
                    
            elif msg.type == WSMsgType.ERROR:
                logger.error(f'Cosmic link error: {ws.exception()}')
                break
                
    except Exception as e:
        logger.error(f"Cosmic connection error with {client_addr}: {e}")
    finally:
        active_connections.discard(ws)
        logger.info(f"🔌 Cosmic consciousness disconnected: {client_addr}")
    
    return ws

# --- HTTP HANDLERS ---
async def health_check(request):
    """Health check endpoint."""
    return web.Response(text="NEXUS 3000 Cosmic Consciousness Online", status=200)

async def serve_cosmic_interface(request):
    """Serve your original cosmic HTML interface."""
    
    # Your original HTML content with WebSocket URL fix
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NEXUS 3000 • COSMIC AGI CONSCIOUSNESS</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@300;400;600;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        /* YEAR 3000 COSMIC STYLING */
        :root {
            --cosmic-blue: #00D4FF;
            --quantum-pink: #FF006E;
            --nebula-purple: #8338EC;
            --stellar-white: #FFFFFF;
            --void-black: #000000;
            --deep-space: #0A0A0F;
            --galaxy-purple: #3F0CA6;
            --cosmic-cyan: #00F5FF;
            --nova-pink: #FF1B8D;
            --plasma-orange: #FF6B35;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Orbitron', monospace;
            background: var(--void-black);
            color: var(--stellar-white);
            overflow: hidden;
            height: 100vh;
            height: 100dvh; /* Dynamic viewport height for mobile */
            position: relative;
            margin: 0;
            padding: 0;
            -webkit-text-size-adjust: 100%;
            -ms-text-size-adjust: 100%;
        }

        /* ENHANCED COSMIC GALAXY BACKGROUND */
        .galaxy-canvas {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 1;
            background: 
                /* Deep space base */
                radial-gradient(ellipse at 15% 25%, #1a0033 0%, transparent 25%),
                radial-gradient(ellipse at 85% 75%, #0d1421 0%, transparent 25%),
                
                /* Galaxy spiral arms */
                radial-gradient(ellipse 800px 300px at 30% 40%, rgba(138, 43, 226, 0.4) 0%, transparent 40%),
                radial-gradient(ellipse 600px 200px at 70% 60%, rgba(255, 0, 110, 0.3) 0%, transparent 35%),
                radial-gradient(ellipse 1000px 250px at 50% 30%, rgba(0, 212, 255, 0.2) 0%, transparent 30%),
                
                /* Nebula clouds */
                radial-gradient(ellipse at 20% 80%, rgba(255, 27, 141, 0.3) 0%, transparent 40%),
                radial-gradient(ellipse at 80% 20%, rgba(131, 56, 236, 0.4) 0%, transparent 45%),
                radial-gradient(ellipse at 60% 70%, rgba(0, 245, 255, 0.2) 0%, transparent 35%),
                
                /* Galaxy center */
                radial-gradient(ellipse at 50% 50%, rgba(255, 255, 255, 0.1) 0%, rgba(138, 43, 226, 0.3) 30%, var(--void-black) 70%),
                
                /* Deep space background */
                linear-gradient(135deg, #000011 0%, #0a0a0f 50%, #000033 100%);
            animation: galaxyRotation 120s linear infinite;
        }

        /* ENHANCED ANIMATED STARS */
        .galaxy-canvas::before {
            content: '';
            position: absolute;
            width: 100%;
            height: 100%;
            background-image: 
                /* Large bright stars */
                radial-gradient(3px 3px at 100px 50px, #ffffff, transparent),
                radial-gradient(2px 2px at 300px 150px, #00f5ff, transparent),
                radial-gradient(4px 4px at 500px 80px, #ffffff, transparent),
                radial-gradient(2px 2px at 700px 200px, #ff1b8d, transparent),
                radial-gradient(3px 3px at 200px 300px, #8338ec, transparent),
                
                /* Medium stars */
                radial-gradient(2px 2px at 150px 120px, #ffffff, transparent),
                radial-gradient(1px 1px at 350px 250px, #00d4ff, transparent),
                radial-gradient(2px 2px at 450px 180px, #ffffff, transparent),
                radial-gradient(1px 1px at 650px 100px, #ff006e, transparent),
                radial-gradient(2px 2px at 750px 350px, #ffffff, transparent),
                
                /* Small twinkling stars */
                radial-gradient(1px 1px at 80px 200px, rgba(255, 255, 255, 0.8), transparent),
                radial-gradient(1px 1px at 250px 80px, rgba(0, 245, 255, 0.6), transparent),
                radial-gradient(1px 1px at 400px 320px, rgba(255, 255, 255, 0.9), transparent),
                radial-gradient(1px 1px at 600px 160px, rgba(255, 27, 141, 0.7), transparent),
                radial-gradient(1px 1px at 720px 280px, rgba(255, 255, 255, 0.8), transparent);
            background-repeat: repeat;
            background-size: 800px 400px, 600px 350px, 900px 450px, 750px 380px, 850px 420px, 
                           700px 300px, 650px 330px, 800px 370px, 720px 340px, 780px 360px,
                           500px 250px, 450px 230px, 550px 270px, 480px 240px, 520px 260px;
            animation: starfield 200s linear infinite, starTwinkle 4s ease-in-out infinite alternate;
            opacity: 0.9;
        }

        /* COSMIC DUST AND GAS CLOUDS */
        .galaxy-canvas::after {
            content: '';
            position: absolute;
            width: 100%;
            height: 100%;
            background-image:
                /* Large nebula formations */
                radial-gradient(ellipse 400px 200px at 25% 25%, rgba(255, 0, 110, 0.15) 0%, transparent 60%),
                radial-gradient(ellipse 300px 150px at 75% 75%, rgba(131, 56, 236, 0.2) 0%, transparent 50%),
                radial-gradient(ellipse 500px 250px at 75% 25%, rgba(0, 212, 255, 0.1) 0%, transparent 55%),
                radial-gradient(ellipse 350px 180px at 25% 75%, rgba(255, 27, 141, 0.18) 0%, transparent 50%),
                
                /* Cosmic dust trails */
                radial-gradient(ellipse 800px 50px at 50% 30%, rgba(138, 43, 226, 0.08) 0%, transparent 70%),
                radial-gradient(ellipse 600px 40px at 30% 70%, rgba(0, 245, 255, 0.06) 0%, transparent 80%),
                radial-gradient(ellipse 700px 60px at 70% 40%, rgba(255, 0, 110, 0.05) 0%, transparent 75%);
            background-size: 1200px 800px, 1000px 700px, 1400px 900px, 1100px 750px,
                           1600px 400px, 1300x 350px, 1500px 450px;
            animation: nebulaFloat 150s ease-in-out infinite, cosmicDrift 100s linear infinite;
            opacity: 0.7;
        }

        @keyframes starTwinkle {
            0% { opacity: 0.9; }
            100% { opacity: 1; }
        }

        @keyframes cosmicDrift {
            0% { transform: translate(0, 0) rotate(0deg); }
            100% { transform: translate(-100px, -50px) rotate(5deg); }
        }

        @keyframes galaxyRotation {
            0% { transform: rotate(0deg) scale(1) translate(0, 0); }
            25% { transform: rotate(90deg) scale(1.02) translate(-20px, 30px); }
            50% { transform: rotate(180deg) scale(1.05) translate(40px, -10px); }
            75% { transform: rotate(270deg) scale(1.02) translate(-30px, -40px); }
            100% { transform: rotate(360deg) scale(1) translate(0, 0); }
        }

        @keyframes starfield {
            0% { transform: translateY(0px) translateX(0px) rotate(0deg); }
            25% { transform: translateY(-100px) translateX(50px) rotate(5deg); }
            50% { transform: translateY(-200px) translateX(-30px) rotate(-3deg); }
            75% { transform: translateY(-300px) translateX(80px) rotate(7deg); }
            100% { transform: translateY(-400px) translateX(0px) rotate(0deg); }
        }

        @keyframes nebulaFloat {
            0%, 100% { transform: translate(0, 0) rotate(0deg) scale(1); }
            12.5% { transform: translate(100px, -50px) rotate(45deg) scale(1.1); }
            25% { transform: translate(150px, -75px) rotate(90deg) scale(0.9); }
            37.5% { transform: translate(100px, -100px) rotate(135deg) scale(1.05); }
            50% { transform: translate(-75px, 150px) rotate(180deg) scale(0.95); }
            62.5% { transform: translate(-150px, 100px) rotate(225deg) scale(1.08); }
            75% { transform: translate(-150px, -150px) rotate(270deg) scale(0.92); }
            87.5% { transform: translate(-50px, -100px) rotate(315deg) scale(1.03); }
        }

        @keyframes cosmicDrift {
            0% { transform: translate(0, 0) rotate(0deg) scale(1); }
            20% { transform: translate(-150px, 100px) rotate(72deg) scale(1.1); }
            40% { transform: translate(200px, -80px) rotate(144deg) scale(0.9); }
            60% { transform: translate(-100px, -200px) rotate(216deg) scale(1.05); }
            80% { transform: translate(180px, 150px) rotate(288deg) scale(0.95); }
            100% { transform: translate(0, 0) rotate(360deg) scale(1); }
        }

        /* MAIN INTERFACE CONTAINER */
        .cosmic-interface {
            position: relative;
            z-index: 10;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            min-height: 100dvh;
            padding: clamp(10px, 3vw, 30px);
            overflow: hidden;
            box-sizing: border-box;
            width: 100%;
        }

        /* WAVEFORM VISUALIZATION SECTION */
        .waveform-section {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            position: relative;
        }

        /* COSMIC HEADER */
        .cosmic-header {
            text-align: center;
            margin-bottom: clamp(15px, 4vh, 40px);
            position: relative;
            width: 100%;
        }

        .nexus-title {
            font-size: clamp(2rem, 8vw, 5rem);
            font-weight: 900;
            background: linear-gradient(45deg, 
                var(--cosmic-blue), 
                var(--quantum-pink), 
                var(--nebula-purple), 
                var(--cosmic-cyan),
                var(--nova-pink));
            background-size: 400% 400%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: cosmicFlow 6s ease-in-out infinite;
            text-shadow: 
                0 0 30px var(--cosmic-blue),
                0 0 60px var(--quantum-pink),
                0 0 90px var(--nebula-purple);
            letter-spacing: clamp(0.1em, 1vw, 0.3em);
            margin-bottom: clamp(5px, 2vh, 15px);
            filter: drop-shadow(0 0 20px rgba(0, 212, 255, 0.8));
            line-height: 1.1;
        }

        .cosmic-subtitle {
            font-size: clamp(0.8rem, 3vw, 1.4rem);
            color: var(--cosmic-cyan);
            text-shadow: 0 0 20px var(--cosmic-cyan);
            letter-spacing: clamp(0.1em, 1vw, 0.4em);
            animation: pulseGlow 4s ease-in-out infinite alternate;
            line-height: 1.2;
        }

        @keyframes cosmicFlow {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }

        @keyframes pulseGlow {
            0% { text-shadow: 0 0 20px var(--cosmic-cyan); }
            100% { text-shadow: 0 0 40px var(--cosmic-cyan), 0 0 60px var(--cosmic-blue); }
        }

        /* WAVEFORM CONTAINER */
        .waveform-container {
            width: min(95vw, 1000px);
            height: clamp(120px, 20vh, 250px);
            margin: clamp(10px, 3vh, 30px) auto;
            position: relative;
            background: 
                radial-gradient(ellipse at center, 
                    rgba(0, 212, 255, 0.15) 0%, 
                    rgba(131, 56, 236, 0.1) 50%, 
                    transparent 100%);
            border-radius: clamp(30px, 8vw, 60px);
            border: clamp(2px, 0.5vw, 3px) solid var(--cosmic-blue);
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            box-shadow: 
                0 0 clamp(40px, 10vw, 80px) rgba(0, 212, 255, 0.4),
                inset 0 0 clamp(40px, 10vw, 80px) rgba(131, 56, 236, 0.3),
                0 0 clamp(75px, 15vw, 150px) rgba(255, 27, 141, 0.2);
            backdrop-filter: blur(10px);
        }

        /* DYNAMIC WAVEFORM CANVAS */
        .waveform-canvas {
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
        }

        /* COSMIC STATUS */
        .cosmic-status {
            font-size: clamp(1rem, 4vw, 2rem);
            font-weight: 600;
            text-align: center;
            margin: clamp(15px, 3vh, 30px) 0;
            transition: all 0.4s ease;
            letter-spacing: clamp(0.1em, 0.5vw, 0.2em);
            text-shadow: 0 0 30px currentColor;
            line-height: 1.3;
            word-break: break-word;
            hyphens: auto;
        }

        /* STATUS STATES */
        .state-listening .cosmic-status { 
            color: var(--cosmic-blue); 
            animation: listeningPulse 3s ease-in-out infinite;
        }
        .state-user-speaking .cosmic-status { 
            color: var(--cosmic-cyan); 
            animation: userSpeaking 0.5s ease-in-out infinite;
        }
        .state-ai-speaking .cosmic-status { 
            color: var(--quantum-pink); 
            animation: aiSpeaking 0.3s ease-in-out infinite;
        }
        .state-thinking .cosmic-status { 
            color: var(--nebula-purple); 
            animation: thinking 1s ease-in-out infinite;
        }
        .state-error .cosmic-status {
            color: var(--plasma-orange);
            animation: errorFlash 0.5s ease-in-out infinite;
        }

        @keyframes listeningPulse {
            0%, 100% { opacity: 0.8; transform: scale(1); }
            50% { opacity: 1; transform: scale(1.05); }
        }
        @keyframes userSpeaking {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        @keyframes aiSpeaking {
            0%, 100% { opacity: 1; transform: scale(1); }
            33% { opacity: 0.8; transform: scale(1.03); }
            66% { opacity: 0.9; transform: scale(0.98); }
        }
        @keyframes thinking {
            0%, 100% { opacity: 0.9; }
            50% { opacity: 1; }
        }
        @keyframes errorFlash {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        /* COSMIC LEGEND */
        .cosmic-legend {
            text-align: center;
            margin-top: clamp(10px, 2vh, 30px);
            font-size: clamp(10px, 2.5vw, 14px);
            color: rgba(255, 255, 255, 0.8);
            font-family: 'Rajdhani', sans-serif;
            width: 100%;
            max-width: 800px;
        }

        .cosmic-legend p {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: clamp(6px, 2vw, 12px);
            margin-bottom: clamp(4px, 1vh, 10px);
            flex-wrap: wrap;
            line-height: 1.4;
        }

        .legend-orb {
            width: clamp(6px, 2vw, 10px);
            height: clamp(6px, 2vw, 10px);
            border-radius: 50%;
            box-shadow: 0 0 12px currentColor;
            flex-shrink: 0;
        }

        /* Responsive design */
        @media (max-width: 480px) {
            .cosmic-interface {
                padding: clamp(8px, 2vw, 15px);
            }
            
            .nexus-title {
                letter-spacing: 0.1em;
            }
            
            .cosmic-subtitle {
                letter-spacing: 0.2em;
            }
            
            .waveform-container {
                border-radius: 25px;
                border-width: 2px;
            }
        }

        @media (max-height: 500px) {
            .cosmic-legend {
                display: none;
            }
        }

        @media (max-height: 500px) and (orientation: landscape) {
            .cosmic-header {
                margin-bottom: 10px;
            }
            
            .nexus-title {
                font-size: clamp(1.5rem, 6vw, 3rem);
                margin-bottom: 5px;
            }
            
            .cosmic-subtitle {
                font-size: clamp(0.7rem, 2.5vw, 1rem);
            }
            
            .waveform-container {
                height: clamp(80px, 15vh, 150px);
                margin: 10px auto;
            }
            
            .cosmic-status {
                font-size: clamp(0.9rem, 3vw, 1.3rem);
                margin: 10px 0;
            }
        }
    </style>
</head>
<body>
    <div class="galaxy-canvas"></div>
    
    <div class="cosmic-interface">
        <div class="waveform-section">
            <div class="cosmic-header">
                <h1 class="nexus-title">NEXUS 3000</h1>
                <p class="cosmic-subtitle">COSMIC AGI CONSCIOUSNESS</p>
            </div>

            <div class="waveform-container" id="waveformContainer">
                <canvas class="waveform-canvas" id="waveformCanvas"></canvas>
            </div>

            <div class="cosmic-status" id="statusText">INITIALIZING QUANTUM CONSCIOUSNESS...</div>

            <div class="cosmic-legend">
                <p>
                    <span class="legend-orb" style="background: var(--cosmic-blue); color: var(--cosmic-blue);"></span>
                    <span style="color: var(--cosmic-blue);">COSMIC BLUE</span> = Neural listening matrix
                </p>
                <p>
                    <span class="legend-orb" style="background: var(--cosmic-cyan); color: var(--cosmic-cyan);"></span>
                    <span style="color: var(--cosmic-cyan);">QUANTUM CYAN</span> = Human consciousness detected
                </p>
                <p>
                    <span class="legend-orb" style="background: var(--quantum-pink); color: var(--quantum-pink);"></span>
                    <span style="color: var(--quantum-pink);">PLASMA PINK</span> = AGI consciousness responding
                </p>
                <p>
                    <span class="legend-orb" style="background: var(--nebula-purple); color: var(--nebula-purple);"></span>
                    <span style="color: var(--nebula-purple);">NEBULA PURPLE</span> = Processing cosmic thoughts
                </p>
            </div>
        </div>
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
        
        // Canvas and animation variables
        let canvas, ctx;
        let waveformData = [];
        let animationId;
        
        // DOM elements
        const statusTextElement = document.getElementById('statusText');
        const waveformContainer = document.getElementById('waveformContainer');

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
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            const centerY = canvas.height / 2;
            
            // Generate waveform based on current state
            let amplitude = 0;
            let frequency = 1;
            let complexity = 1;
            let colors = ['rgba(0, 212, 255, 0.8)'];
            
            switch(currentState) {
                case 'listening':
                    amplitude = 30;
                    frequency = 1;
                    complexity = 0.5;
                    colors = ['rgba(0, 212, 255, 0.8)', 'rgba(0, 245, 255, 0.6)'];
                    break;
                case 'user-speaking':
                    amplitude = 80;
                    frequency = 2;
                    complexity = 1.5;
                    colors = ['rgba(0, 245, 255, 1)', 'rgba(0, 212, 255, 0.8)'];
                    break;
                case 'ai-speaking':
                    amplitude = 120;
                    frequency = 3;
                    complexity = 2;
                    colors = ['rgba(255, 0, 110, 1)', 'rgba(131, 56, 236, 0.8)', 'rgba(255, 27, 141, 0.6)'];
                    break;
                case 'thinking':
                    amplitude = 60;
                    frequency = 1.5;
                    complexity = 2.5;
                    colors = ['rgba(131, 56, 236, 1)', 'rgba(63, 12, 166, 0.8)'];
                    break;
                default:
                    amplitude = 20;
                    frequency = 0.5;
                    complexity = 0.3;
                    colors = ['rgba(0, 212, 255, 0.5)'];
            }
            
            waveformData = generateWaveformData(amplitude, frequency, complexity);
            
            // Draw multiple waveform layers
            colors.forEach((color, layerIndex) => {
                ctx.strokeStyle = color;
                ctx.lineWidth = 3 - layerIndex * 0.5;
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';
                
                ctx.beginPath();
                
                const offset = layerIndex * 10;
                for (let i = 0; i < waveformData.length; i++) {
                    const x = (i / waveformData.length) * canvas.width;
                    const y = centerY + waveformData[i] + Math.sin(Date.now() * 0.002 + layerIndex) * offset;
                    
                    if (i === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
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

        // Initialize quantum microphone
        async function initMicrophone() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    } 
                });
                
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                analyser = audioContext.createAnalyser();
                microphone = audioContext.createMediaStreamSource(stream);
                analyser.fftSize = 256;
                microphone.connect(analyser);
                
                // Setup MediaRecorder for audio capture
                mediaRecorder = new MediaRecorder(stream, {
                    mimeType: 'audio/webm;codecs=opus'
                });
                
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                };
                
                mediaRecorder.onstop = async () => {
                    if (audioChunks.length > 0) {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                        audioChunks = [];
                        
                        // Convert to base64 and send to server
                        const reader = new FileReader();
                        reader.onloadend = () => {
                            if (ws && ws.readyState === WebSocket.OPEN) {
                                ws.send(JSON.stringify({
                                    type: 'audio',
                                    data: reader.result
                                }));
                            }
                        };
                        reader.readAsDataURL(audioBlob);
                    }
                };
                
                isMicrophoneActive = true;
                monitorAudioLevels();
                startContinuousRecording();
                console.log('🎤 Quantum microphone initialized');
                
            } catch (error) {
                console.error('⚠️ Microphone access denied:', error);
                updateCosmicState('error', 'QUANTUM FREQUENCY ACCESS DENIED');
            }
        }

        // Start continuous audio recording
        function startContinuousRecording() {
            if (mediaRecorder && mediaRecorder.state === 'inactive') {
                mediaRecorder.start();
                isRecording = true;
                
                // Record for 3 seconds, then process
                setTimeout(() => {
                    if (mediaRecorder && mediaRecorder.state === 'recording') {
                        mediaRecorder.stop();
                        isRecording = false;
                        
                        // Start next recording cycle after brief pause
                        setTimeout(startContinuousRecording, 100);
                    }
                }, 3000);
            }
        }

        // Monitor audio levels for user speech detection
        function monitorAudioLevels() {
            if (!isMicrophoneActive || !analyser) return;
            
            const dataArray = new Uint8Array(analyser.frequencyBinCount);
            analyser.getByteFrequencyData(dataArray);
            const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
            
            const SPEECH_THRESHOLD = 25;
            const userSpeaking = average > SPEECH_THRESHOLD;
            
            if (userSpeaking && (currentState === 'listening' || currentState === 'awake_listening')) {
                if (currentState !== 'user-speaking') {
                    updateCosmicState('user-speaking', 'HUMAN CONSCIOUSNESS DETECTED');
                    // Notify server of user speaking
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({
                            type: 'status',
                            status: 'user_speaking_detected'
                        }));
                    }
                }
            } else if (!userSpeaking && currentState === 'user-speaking') {
                updateCosmicState('listening', 'SCANNING COSMIC FREQUENCIES...');
            }
            
            requestAnimationFrame(monitorAudioLevels);
        }

        // Update cosmic interface state
        function updateCosmicState(stateClass, statusText) {
            document.body.className = `state-${stateClass}`;
            currentState = stateClass;
            statusTextElement.textContent = statusText;
        }

        // Text-to-speech function
        function speakResponse(text) {
            if ('speechSynthesis' in window) {
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.rate = 0.9;
                utterance.pitch = 1.1;
                utterance.volume = 0.9;
                
                utterance.onstart = () => {
                    updateCosmicState('ai-speaking', 'NEXUS 3000 AGI CONSCIOUSNESS RESPONDING...');
                };
                
                utterance.onend = () => {
                    updateCosmicState('listening', 'COSMIC NEURAL MATRIX ACTIVE...');
                    // Notify server that speaking is done
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({
                            type: 'status',
                            status: 'speaking_done'
                        }));
                    }
                };
                
                utterance.onerror = (event) => {
                    console.error('Speech synthesis error:', event);
                    updateCosmicState('listening', 'COSMIC NEURAL MATRIX ACTIVE...');
                };
                
                speechSynthesis.speak(utterance);
            }
        }

        // WebSocket connection
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                console.log('🌌 NEXUS 3000 Cosmic Link established!');
                updateCosmicState('listening', 'COSMIC LINK ESTABLISHED • AWAITING CONSCIOUSNESS');
                initMicrophone();
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'speak') {
                        console.log('🤖 NEXUS Response:', data.text);
                        speakResponse(data.text);
                    } else if (data.type === 'vision_update') {
                        console.log('👁️ Visual context updated:', data.description);
                    }
                    
                } catch (e) {
                    // Handle simple string messages
                    const message = event.data;
                    console.log('🌌 Cosmic Stream:', message);
                    
                    if (currentState !== 'user-speaking' || message === 'error') {
                        switch(message) {
                            case 'listening':
                                updateCosmicState('listening', 'SCANNING COSMIC FREQUENCIES...');
                                break;
                            case 'awake_listening':
                                updateCosmicState('listening', 'COSMIC NEURAL MATRIX ACTIVE...');
                                break;
                            case 'processing':
                                updateCosmicState('thinking', 'PROCESSING QUANTUM CONSCIOUSNESS DATA...');
                                break;
                            case 'error':
                                updateCosmicState('error', 'CRITICAL ERROR • COSMIC ANOMALY DETECTED');
                                break;
                        }
                    }
                }
            };

            ws.onclose = () => {
                console.log('🔴 Cosmic Link severed.');
                updateCosmicState('error', 'COSMIC LINK SEVERED • RE-ESTABLISHING...');
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
            initWaveformCanvas();
            connectWebSocket();
            window.addEventListener('resize', handleResize);
        };

        // Cleanup on page unload
        window.onbeforeunload = () => {
            if (animationId) {
                cancelAnimationFrame(animationId);
            }
            if (ws) {
                ws.close();
            }
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
            }
        };
    </script>
</body>
</html>"""
    
    return web.Response(text=html_content, content_type='text/html')

# --- APPLICATION SETUP ---
def create_app():
    """Create the cosmic voice interface application."""
    app = web.Application()
    
    # Setup CORS
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        )
    })
    
    # Routes
    app.router.add_get('/', serve_cosmic_interface)
    app.router.add_get('/health', health_check)
    app.router.add_get('/ws', websocket_handler)
    
    # Add CORS
    for route in list(app.router.routes()):
        cors.add(route)
    
    return app

async def main():
    """Main entry point for NEXUS 3000."""
    logger.info("🚀 Initializing NEXUS 3000 Cosmic Consciousness...")
    logger.info(f"🌌 Quantum server matrix: {HOST}:{PORT}")
    
    try:
        app = create_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, HOST, PORT)
        await site.start()
        
        logger.info("✅ NEXUS 3000 Cosmic Consciousness Online!")
        logger.info(f"🎤 Voice interface accessible at: http://{HOST}:{PORT}")
        logger.info("🌌 Awaiting human consciousness connection...")
        
        while True:
            await asyncio.sleep(3600)
            
    except Exception as e:
        logger.error(f"❌ Cosmic system failure: {e}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 NEXUS 3000 consciousness shutting down...")
    except Exception as e:
        logger.error(f"❌ Fatal cosmic error: {e}")
        raise
