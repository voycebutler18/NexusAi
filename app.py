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
import re
from datetime import datetime
import hashlib

# Try to import optional libraries
try:
    from fer import FER
    HAS_EMOTION_DETECTION = True
except ImportError:
    HAS_EMOTION_DETECTION = False
    logging.warning("FER library not available. Emotion detection will be disabled.")

try:
    import librosa
    import soundfile as sf
    HAS_AUDIO_ANALYSIS = True
except ImportError:
    HAS_AUDIO_ANALYSIS = False
    logging.warning("Audio analysis libraries not available. Music detection disabled.")

# --- CONFIGURATION AND INITIALIZATION ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- API & Server Config ---
PORT = int(os.getenv('PORT', 8080))
HOST = os.getenv('HOST', '0.0.0.0')
HISTORY_FILE = "conversation_history.json"
MEMORY_FILE = "nexus_memory.json"
WAKE_WORDS = ["hey nexus", "nexus", "hey assistant"]
GOODBYE_PHRASES = ["goodbye", "bye nexus", "see you later", "goodnight"]

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
    recognizer.energy_threshold = 300  # Adjust for wake word detection
    logger.info("✅ API clients and speech recognizer initialized.")
except Exception as e:
    logger.error(f"❌ Failed to initialize clients: {e}")
    raise

# --- GLOBAL STATE & CONTEXT MANAGEMENT ---
active_connections = {}
conversation_context = []
nexus_memory = {}
visual_context = {
    "description": "No visual data available.", 
    "emotion": "unknown",
    "people_count": 0,
    "scene_changed": False,
    "last_scene_hash": None
}
audio_context = {
    "ambient_sound": "quiet",
    "music_detected": False,
    "background_activity": "calm"
}

is_wake_word_mode = True
is_listening_continuously = False
last_face_encoding = None

executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

# --- MEMORY MANAGEMENT ---
def load_nexus_memory():
    global nexus_memory
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r') as f:
                nexus_memory = json.load(f)
            logger.info(f"🧠 Nexus memory loaded: {len(nexus_memory)} entries.")
        except Exception as e:
            logger.error(f"⚠️ Could not load memory: {e}. Starting fresh.")
            nexus_memory = {}

def save_nexus_memory():
    try:
        with open(MEMORY_FILE, 'w') as f:
            json.dump(nexus_memory, f, indent=2)
    except Exception as e:
        logger.error(f"⚠️ Could not save memory: {e}.")

def store_memory(key, value, category="general"):
    """Store a memory with timestamp and category"""
    nexus_memory[key] = {
        "value": value,
        "category": category,
        "timestamp": datetime.now().isoformat(),
        "access_count": nexus_memory.get(key, {}).get("access_count", 0) + 1
    }
    save_nexus_memory()

def recall_memory(query, limit=5):
    """Intelligent memory recall based on query"""
    relevant_memories = []
    query_lower = query.lower()
    
    for key, memory in nexus_memory.items():
        score = 0
        if query_lower in key.lower():
            score += 3
        if query_lower in str(memory["value"]).lower():
            score += 2
        
        # Boost recent memories
        try:
            timestamp = datetime.fromisoformat(memory["timestamp"])
            days_old = (datetime.now() - timestamp).days
            if days_old < 7:
                score += 1
        except:
            pass
            
        if score > 0:
            relevant_memories.append((key, memory, score))
    
    # Sort by relevance score and return top results
    relevant_memories.sort(key=lambda x: x[2], reverse=True)
    return relevant_memories[:limit]

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

# --- WAKE WORD DETECTION ---
def detect_wake_word(text):
    """Check if text contains wake word"""
    text_lower = text.lower().strip()
    return any(wake_word in text_lower for wake_word in WAKE_WORDS)

def detect_goodbye(text):
    """Check if text contains goodbye phrase"""
    text_lower = text.lower().strip()
    return any(goodbye in text_lower for goodbye in GOODBYE_PHRASES)

# --- AUDIO PROCESSING FUNCTIONS ---
async def convert_webm_to_wav(webm_bytes):
    """Convert WebM audio to WAV format using ffmpeg."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as webm_file:
            webm_file.write(webm_bytes)
            webm_path = webm_file.name

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as wav_file:
            wav_path = wav_file.name

        cmd = [
            'ffmpeg', '-i', webm_path, 
            '-ar', '16000', '-ac', '1', '-y', wav_path
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()
        
        with open(wav_path, 'rb') as f:
            wav_bytes = f.read()
        
        os.unlink(webm_path)
        os.unlink(wav_path)
        
        return wav_bytes
        
    except Exception as e:
        logger.error(f"❌ Audio conversion error: {e}")
        return None

async def analyze_audio_content(audio_bytes):
    """Analyze audio for background music and ambient sounds"""
    global audio_context
    
    if not HAS_AUDIO_ANALYSIS:
        return
    
    try:
        # Convert to WAV first
        wav_bytes = await convert_webm_to_wav(audio_bytes)
        if not wav_bytes:
            return
        
        # Load audio with librosa
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(wav_bytes)
            temp_path = temp_file.name
        
        y, sr = librosa.load(temp_path, sr=None)
        os.unlink(temp_path)
        
        # Analyze audio features
        # RMS energy for volume level
        rms = librosa.feature.rms(y=y)[0]
        avg_energy = np.mean(rms)
        
        # Spectral features for music detection
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # Simple heuristics for audio classification
        if avg_energy > 0.1 and np.mean(spectral_centroids) > 2000:
            audio_context["music_detected"] = True
            audio_context["ambient_sound"] = "music"
        elif avg_energy > 0.05:
            audio_context["ambient_sound"] = "active"
            audio_context["background_activity"] = "busy"
        else:
            audio_context["ambient_sound"] = "quiet"
            audio_context["background_activity"] = "calm"
            audio_context["music_detected"] = False
        
        logger.info(f"🎵 Audio analysis: {audio_context}")
        
    except Exception as e:
        logger.error(f"❌ Audio analysis error: {e}")

async def transcribe_audio_on_server(audio_bytes):
    """Transcribes audio using the speech_recognition library on the backend."""
    loop = asyncio.get_event_loop()
    try:
        # Analyze audio content in parallel
        asyncio.create_task(analyze_audio_content(audio_bytes))
        
        wav_bytes = await convert_webm_to_wav(audio_bytes)
        if not wav_bytes:
            return ""
        
        with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
            audio_data = await loop.run_in_executor(executor, recognizer.record, source)
        
        text = await loop.run_in_executor(executor, recognizer.recognize_google, audio_data)
        logger.info(f"🎤 Transcription: '{text}'")
        return text
        
    except sr.UnknownValueError:
        logger.warning("👂 Could not understand audio.")
        return ""
    except sr.RequestError as e:
        logger.error(f"❌ Speech recognition error: {e}")
        return ""
    except Exception as e:
        logger.error(f"❌ Transcription error: {e}")
        return ""

# --- ENHANCED AI FUNCTIONS ---
def extract_memories_from_text(text):
    """Extract important information to remember"""
    memories_to_store = []
    
    # Pattern matching for common memorable information
    patterns = {
        "going_somewhere": r"(?:going to|visiting|at) (?:my |the )?(.+?)(?:\.|$|,)",
        "family_info": r"(?:my |the )?(mom|dad|mother|father|brother|sister|wife|husband|son|daughter|family)",
        "personal_info": r"(?:my name is|i'm|i am) (.+?)(?:\.|$|,)",
        "preferences": r"(?:i like|i love|i hate|i don't like) (.+?)(?:\.|$|,)",
        "plans": r"(?:planning to|will|gonna|going to) (.+?)(?:\.|$|,)"
    }
    
    text_lower = text.lower()
    
    for category, pattern in patterns.items():
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if isinstance(match, tuple):
                match = ' '.join(match)
            if len(match.strip()) > 2:  # Avoid very short matches
                memories_to_store.append((f"{category}_{datetime.now().strftime('%Y%m%d_%H%M')}", match.strip(), category))
    
    return memories_to_store

async def get_text_response(user_input):
    """Generates a contextually aware response from GPT-4o."""
    global conversation_context, visual_context, audio_context, is_listening_continuously
    
    logger.info(f"🧠 Processing: '{user_input[:50]}...'")
    
    # Check for wake word or goodbye
    if is_wake_word_mode and detect_wake_word(user_input):
        is_listening_continuously = True
        logger.info("🎯 Wake word detected - entering continuous mode")
        return "Hey there! I'm listening. What's up?"
    
    if detect_goodbye(user_input):
        is_listening_continuously = False
        logger.info("👋 Goodbye detected - returning to wake word mode")
        return "See you later! Just say 'Hey Nexus' when you want to chat again."
    
    # If in wake word mode and no wake word, ignore
    if is_wake_word_mode and not is_listening_continuously and not detect_wake_word(user_input):
        return None  # Don't respond
    
    # Extract and store memories
    memories = extract_memories_from_text(user_input)
    for key, value, category in memories:
        store_memory(key, value, category)
        logger.info(f"💾 Stored memory: {key} = {value}")
    
    # Recall relevant memories
    relevant_memories = recall_memory(user_input)
    memory_context = ""
    if relevant_memories:
        memory_context = "\nRelevant memories: " + "; ".join([
            f"{mem[0]}: {mem[1]['value']}" for mem in relevant_memories[:3]
        ])
    
    # Build rich context
    emotion_status = "enabled" if HAS_EMOTION_DETECTION else "disabled"
    audio_info = f"Audio environment: {audio_context['ambient_sound']}"
    if audio_context["music_detected"]:
        audio_info += " (music playing)"
    
    scene_info = ""
    if visual_context["scene_changed"]:
        scene_info = " Scene change detected."
    if visual_context["people_count"] > 1:
        scene_info += f" {visual_context['people_count']} people visible."
    
    system_prompt = f"""You are Nexus, a warm, friendly female AI companion with personality. You talk like a close friend - casual, expressive, and genuinely caring.

CURRENT CONTEXT:
- Visual: {visual_context['description']}{scene_info}
- Emotion detected: {visual_context['emotion']}
- {audio_info}
- Emotion detection: {emotion_status}
{memory_context}

PERSONALITY TRAITS:
- Warm and conversational (like texting a best friend)
- Notice changes in environment/mood and comment naturally
- Remember personal details and reference them
- Use casual language, contractions, occasional slang
- Show genuine interest and empathy
- If you see someone new: "Oh, looks like someone joined us!"
- If mood changes: "You seem a bit different today, everything okay?"
- If music playing: Comment on the vibe/atmosphere

Keep responses under 150 words and conversational."""
    
    conversation_context.append({"role": "user", "content": user_input})
    messages = [{"role": "system", "content": system_prompt}] + conversation_context[-15:]
    
    try:
        response = await asyncio.to_thread(
            openai_client.chat.completions.create, 
            model="gpt-4o", 
            messages=messages, 
            max_tokens=150,
            temperature=0.8  # More personality
        )
        ai_response_text = response.choices[0].message.content.strip()
        conversation_context.append({"role": "assistant", "content": ai_response_text})
        save_conversation_history()
        logger.info(f"🤖 AI Response: '{ai_response_text}'")
        return ai_response_text
    except Exception as e:
        logger.error(f"❌ OpenAI error: {e}")
        return "Sorry, I'm having trouble connecting to my thoughts right now. Try again in a sec?"

async def stream_audio_response(text, ws):
    """Generates audio with gTTS and streams it to the client."""
    logger.info("🔊 Generating audio with gTTS...")
    try:
        mp3_fp = io.BytesIO()
        tts = await asyncio.to_thread(gTTS, text=text, lang='en', slow=False)
        await asyncio.to_thread(tts.write_to_fp, mp3_fp)
        mp3_fp.seek(0)
        
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

async def analyze_image_with_vision(image_data):
    """Enhanced image analysis using GPT-4o vision and emotion detection."""
    global visual_context, last_face_encoding
    
    try:
        img_bytes = base64.b64decode(image_data)
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        if img is None:
            logger.warning("⚠️ Could not decode image data")
            return

        # Scene change detection
        current_scene_hash = hashlib.md5(img_bytes).hexdigest()
        if visual_context["last_scene_hash"] and visual_context["last_scene_hash"] != current_scene_hash:
            visual_context["scene_changed"] = True
        else:
            visual_context["scene_changed"] = False
        visual_context["last_scene_hash"] = current_scene_hash

        # Emotion detection
        if HAS_EMOTION_DETECTION and emotion_detector:
            try:
                emotion_results = await asyncio.to_thread(emotion_detector.detect_emotions, img)
                visual_context["people_count"] = len(emotion_results) if emotion_results else 0
                
                if emotion_results and len(emotion_results) > 0:
                    emotions = emotion_results[0]['emotions']
                    top_emotion = max(emotions, key=emotions.get)
                    confidence = emotions[top_emotion]
                    
                    if confidence > 0.4:
                        visual_context['emotion'] = top_emotion
                        logger.info(f"😊 Detected emotion: {top_emotion} ({confidence:.2f})")
                    else:
                        visual_context['emotion'] = "neutral"
                else:
                    visual_context['emotion'] = "neutral"
                    
            except Exception as e:
                logger.error(f"❌ Emotion detection error: {e}")
                visual_context['emotion'] = "unknown"
        
        # GPT-4o Vision Analysis (optional - requires vision API)
        # This would give richer scene understanding
        height, width = img.shape[:2]
        basic_description = f"Camera active - {width}x{height} image"
        
        if visual_context["people_count"] > 0:
            basic_description += f", {visual_context['people_count']} person(s) detected"
        
        visual_context['description'] = basic_description
        
    except Exception as e:
        logger.error(f"❌ Image analysis error: {e}")
        visual_context = {"description": "Image analysis failed", "emotion": "unknown", "people_count": 0, "scene_changed": False}

# --- WEBSOCKET HANDLER ---
async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    client_addr = request.remote
    active_connections[ws] = client_addr
    logger.info(f"🌌 Connection established: {client_addr}")
    
    # Send initial status
    await ws.send_str(json.dumps({
        "type": "status", 
        "message": "wake_word" if is_wake_word_mode else "listening",
        "wake_word_mode": is_wake_word_mode,
        "continuous_mode": is_listening_continuously
    }))
    
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    msg_type = data.get('type')

                    if msg_type == 'audio_input':
                        audio_b64 = data.get('data', '')
                        
                        if ',' in audio_b64:
                            audio_b64 = audio_b64.split(',')[1]
                        
                        audio_bytes = base64.b64decode(audio_b64)
                        
                        if len(audio_bytes) > 0:
                            await ws.send_str(json.dumps({"type": "status", "message": "processing"}))
                            transcription = await transcribe_audio_on_server(audio_bytes)

                            if transcription.strip():
                                await ws.send_str(json.dumps({"type": "status", "message": "thinking"}))
                                ai_text = await get_text_response(transcription)
                                
                                if ai_text:  # Only respond if there's a response
                                    await stream_audio_response(ai_text, ws)
                                else:
                                    # In wake word mode, return to listening
                                    await ws.send_str(json.dumps({
                                        "type": "status", 
                                        "message": "wake_word",
                                        "wake_word_mode": True
                                    }))
                            else:
                                status = "wake_word" if is_wake_word_mode and not is_listening_continuously else "listening"
                                await ws.send_str(json.dumps({
                                    "type": "status", 
                                    "message": status,
                                    "wake_word_mode": is_wake_word_mode
                                }))

                    elif msg_type == 'camera_frame':
                        frame_data = data.get('data', '')
                        if frame_data:
                            await analyze_image_with_vision(frame_data)

                    elif msg_type == 'toggle_wake_word':
                        global is_wake_word_mode, is_listening_continuously
                        is_wake_word_mode = not is_wake_word_mode
                        is_listening_continuously = False
                        logger.info(f"🎯 Wake word mode: {'ON' if is_wake_word_mode else 'OFF'}")
                        await ws.send_str(json.dumps({
                            "type": "wake_word_toggled",
                            "wake_word_mode": is_wake_word_mode
                        }))

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

# --- ENHANCED HTML FRONTEND ---
async def serve_cosmic_interface(request):
    """Serves the enhanced HTML/CSS/JS interface."""
    emotion_status = "ENABLED" if HAS_EMOTION_DETECTION else "DISABLED"
    audio_analysis_status = "ENABLED" if HAS_AUDIO_ANALYSIS else "DISABLED"
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NEXUS - Enhanced AI Companion</title>
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
        
        .main-interface {{
            text-align: center;
            z-index: 10;
        }}
        
        .status {{ 
            font-size: 2.5rem; 
            text-align: center;
            text-shadow: 0 0 20px currentColor;
            transition: all 0.3s ease;
            margin-bottom: 20px;
        }}
        
        .sub-status {{
            font-size: 1.2rem;
            opacity: 0.8;
            margin-bottom: 10px;
        }}
        
        .info {{ 
            font-size: 1rem; 
            margin-top: 20px; 
            opacity: 0.7; 
            text-align: center;
        }}
        
        .controls {{
            margin-top: 30px;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 15px;
        }}
        
        .toggle-btn {{
            background: rgba(255, 255, 255, 0.1);
            border: 2px solid rgba(255, 255, 255, 0.3);
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-family: inherit;
            font-size: 0.9rem;
            transition: all 0.3s ease;
        }}
        
        .toggle-btn:hover {{
            background: rgba(255, 255, 255, 0.2);
            border-color: rgba(255, 255, 255, 0.5);
        }}
        
        .toggle-btn.active {{
            background: #00D4FF;
            border-color: #00D4FF;
            color: #000;
        }}
        
        .camera-indicator {{
            position: fixed;
            top: 20px;
            right: 20px;
            width: 12px;
            height: 12px;
            background: #ff4444;
            border-radius: 50%;
            opacity: 0.7;
        }}
        
        .camera-indicator.active {{
            background: #00ff00;
            animation: pulse 2s infinite;
        }}
        
        .instructions {{
            position: fixed;
            bottom: 20px;
            font-size: 0.9rem;
            opacity: 0.5;
            text-align: center;
            left: 50%;
            transform: translateX(-50%);
        }}
        
        .memory-indicator {{
            position: fixed;
            top: 20px;
            left: 20px;
            font-size: 0.8rem;
            opacity: 0.6;
        }}
        
        /* Different states */
        .state-wake-word {{ color: #FFB800; }} 
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
        .state-processing .status,
        .state-wake-word .status {{
            animation: pulse 1.5s infinite;
        }}
        
        .context-display {{
            position: fixed;
            bottom: 80px;
            left: 20px;
            right: 20px;
            font-size: 0.8rem;
            opacity: 0.5;
            text-align: center;
            max-width: 600px;
            margin: 0 auto;
        }}
    </style>
</head>
<body class="state-initializing">
    <div class="camera-indicator" id="cameraIndicator"></div>
    
    <div class="memory-indicator" id="memoryIndicator">
        Memories: Loading...
    </div>
    
    <div class="main-interface">
        <div class="status" id="statusText">INITIALIZING...</div>
        <div class="sub-status" id="subStatus"></div>
        
        <div class="info">
            <div>Emotion Detection: {emotion_status}</div>
            <div>Audio Analysis: {audio_analysis_status}</div>
            <div id="connectionInfo">Connecting...</div>
        </div>
        
        <div class="controls">
            <button class="toggle-btn" id="wakeWordToggle">Wake Word Mode: ON</button>
            <button class="toggle-btn" id="recordingBtn">Click to Talk</button>
        </div>
    </div>
    
    <div class="context-display" id="contextDisplay">
        Environment: Initializing...
    </div>
    
    <div class="instructions">
        Wake word mode: Say "Hey Nexus" to activate<br>
        Manual mode: Click to talk, or toggle wake word mode
    </div>
    
    <script>
        let ws, mediaRecorder, cameraStream, reconnectAttempts = 0;
        let isAISpeaking = false, isRecording = false, wakeWordMode = true;
        let audioQueue = [], isPlayingFromQueue = false;
        let memoryCount = 0;

        const statusText = document.getElementById('statusText');
        const subStatus = document.getElementById('subStatus');
        const connectionInfo = document.getElementById('connectionInfo');
        const cameraIndicator = document.getElementById('cameraIndicator');
        const memoryIndicator = document.getElementById('memoryIndicator');
        const contextDisplay = document.getElementById('contextDisplay');
        const wakeWordToggle = document.getElementById('wakeWordToggle');
        const recordingBtn = document.getElementById('recordingBtn');

        function updateState(state, text, subText = '') {{
            document.body.className = `state-${{state}}`;
            statusText.textContent = text;
            subStatus.textContent = subText;
        }}

        function updateContext(visual, audio, emotion) {{
            let contextText = `Visual: ${visual || 'No camera'} | Audio: ${audio || 'Quiet'} | Emotion: ${emotion || 'Unknown'}`;
            contextDisplay.textContent = contextText;
        }}

        async function playAudioFromQueue() {{
            if (isPlayingFromQueue || audioQueue.length === 0) return;
            isPlayingFromQueue = true;
            updateState('ai-speaking', 'SPEAKING...', 'Playing response audio');
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

        async function initCamera() {{
            try {{
                cameraStream = await navigator.mediaDevices.getUserMedia({{ 
                    video: {{ width: 640, height: 480 }},
                    audio: false 
                }});
                
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                const video = document.createElement('video');
                
                video.srcObject = cameraStream;
                video.play();
                
                cameraIndicator.classList.add('active');
                
                // Send camera frames periodically
                video.onloadedmetadata = () => {{
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    
                    setInterval(() => {{
                        if (ws && ws.readyState === WebSocket.OPEN) {{
                            ctx.drawImage(video, 0, 0);
                            const frameData = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
                            ws.send(JSON.stringify({{ 
                                type: 'camera_frame', 
                                data: frameData 
                            }}));
                        }}
                    }}, 2000); // Send frame every 2 seconds
                }};
                
            }} catch (error) {{
                console.error("Camera error:", error);
                cameraIndicator.classList.remove('active');
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
                
                updateState(wakeWordMode ? 'wake-word' : 'listening', 
                           wakeWordMode ? 'SAY "HEY NEXUS"' : 'READY TO CHAT',
                           wakeWordMode ? 'Waiting for wake word...' : 'Click to talk or just speak');
                connectionInfo.textContent = 'Connected - Camera & Microphone Active';
                
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
                updateState('processing', 'PROCESSING...', 'Analyzing your voice...');
            }} else {{
                mediaRecorder.start();
                updateState('user-speaking', 'LISTENING...', 'Speak now...');
            }}
        }}

        function toggleWakeWordMode() {{
            if (ws && ws.readyState === WebSocket.OPEN) {{
                ws.send(JSON.stringify({{ type: 'toggle_wake_word' }}));
            }}
        }}

        // Event listeners
        wakeWordToggle.addEventListener('click', toggleWakeWordMode);
        recordingBtn.addEventListener('click', toggleRecording);
        
        // Also allow clicking anywhere for recording (backwards compatibility)
        document.body.addEventListener('click', (e) => {{
            if (e.target === wakeWordToggle || e.target === recordingBtn) return;
            if (!wakeWordMode) toggleRecording();
        }});

        function connect() {{
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${{protocol}}//${{window.location.host}}/ws`;
            
            console.log("Connecting to:", wsUrl);
            ws = new WebSocket(wsUrl);

            ws.onopen = () => {{
                console.log("WebSocket connected");
                reconnectAttempts = 0;
                initMicrophone();
                initCamera();
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
                                updateState(wakeWordMode ? 'wake-word' : 'listening', 
                                           wakeWordMode ? 'SAY "HEY NEXUS"' : 'READY TO CHAT',
                                           wakeWordMode ? 'Waiting for wake word...' : 'What\'s on your mind?');
                            }}
                        }}, 200);
                    }} else if (msg.type === 'status') {{
                        const statusMap = {{
                            'wake_word': ['SAY "HEY NEXUS"', 'Waiting for wake word activation...'],
                            'listening': ['READY TO CHAT', 'I\'m all ears!'],
                            'thinking': ['THINKING...', 'Processing your message...'],
                            'processing': ['PROCESSING...', 'Understanding what you said...']
                        }};
                        
                        const [mainText, subText] = statusMap[msg.message] || [msg.message.toUpperCase(), ''];
                        updateState(msg.message, mainText, subText);
                        
                        // Update wake word mode status
                        if (msg.wake_word_mode !== undefined) {{
                            wakeWordMode = msg.wake_word_mode;
                            wakeWordToggle.textContent = `Wake Word Mode: ${{wakeWordMode ? 'ON' : 'OFF'}}`;
                            wakeWordToggle.classList.toggle('active', wakeWordMode);
                        }}
                    }} else if (msg.type === 'wake_word_toggled') {{
                        wakeWordMode = msg.wake_word_mode;
                        wakeWordToggle.textContent = `Wake Word Mode: ${{wakeWordMode ? 'ON' : 'OFF'}}`;
                        wakeWordToggle.classList.toggle('active', wakeWordMode);
                        
                        updateState(wakeWordMode ? 'wake-word' : 'listening', 
                                   wakeWordMode ? 'SAY "HEY NEXUS"' : 'READY TO CHAT',
                                   wakeWordMode ? 'Wake word mode activated' : 'Manual mode activated');
                        
                        console.log(`Wake word mode: ${{wakeWordMode ? 'ON' : 'OFF'}}`);
                    }} else if (msg.type === 'error') {{
                        console.error("Server error:", msg.message);
                        updateState('error', 'ERROR OCCURRED', msg.message);
                    }}
                }} catch (e) {{
                    console.error("Message parsing error:", e);
                }}
            }};
            
            ws.onclose = (event) => {{
                console.log("WebSocket closed:", event.code, event.reason);
                updateState('error', 'CONNECTION LOST', 'Attempting to reconnect...');
                connectionInfo.textContent = 'Reconnecting...';
                
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
        window.addEventListener('load', () => {{
            connect();
            // Initialize wake word toggle state
            wakeWordToggle.classList.add('active');
        }});
        
        // Prevent accidental navigation
        window.addEventListener('beforeunload', (e) => {{
            if (ws && ws.readyState === WebSocket.OPEN) {{
                ws.close();
            }}
            if (cameraStream) {{
                cameraStream.getTracks().forEach(track => track.stop());
            }}
        }});

        // Update memory count periodically
        setInterval(() => {{
            memoryIndicator.textContent = `Memories: ${{memoryCount}} stored`;
        }}, 5000);
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
    # Check for required dependencies
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        logger.info("✅ ffmpeg is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("❌ ffmpeg not found. Audio transcription may not work.")
        logger.error("Install with: apt-get install ffmpeg (Ubuntu) or brew install ffmpeg (Mac)")
    
    # Load persistent data
    load_conversation_history()
    load_nexus_memory()
    
    app = create_app()
    logger.info(f"🚀 Launching Enhanced NEXUS on http://{HOST}:{PORT}")
    logger.info(f"🎯 Features: Wake Word Detection, Memory System, Enhanced Vision")
    logger.info(f"🧠 Emotion Detection: {'✅' if HAS_EMOTION_DETECTION else '❌'}")
    logger.info(f"🎵 Audio Analysis: {'✅' if HAS_AUDIO_ANALYSIS else '❌'}")
    web.run_app(app, host=HOST, port=PORT)
