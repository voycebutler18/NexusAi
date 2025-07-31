"""
NEXUS - AI-Powered Voice-Activated Camera-Aware Virtual Companion
Deployment-ready version for Render - Your 24/7 AI Best Friend
"""

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
import cv2
import numpy as np
import re
from datetime import datetime, timedelta
import hashlib
import uuid

# --- RENDER CONFIGURATION ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Render environment
PORT = int(os.getenv('PORT', 10000))
HOST = '0.0.0.0'

# Use /tmp for temporary files on Render
MEMORY_FILE = "/tmp/nexus_memory.json"
CONVERSATION_FILE = "/tmp/conversations.json"

# NEXUS constants
WAKE_WORDS = ["hey nexus", "nexus", "hey girl", "hey friend"]
GOODBYE_PHRASES = ["goodbye nexus", "bye girl", "see you later", "goodnight"]

# --- NEXUS CORE SYSTEMS ---

class NEXUSMemory:
    """Memory system for NEXUS to remember conversations"""
    
    def __init__(self):
        self.memories = {}
        self.conversations = []
        self.load_data()
    
    def load_data(self):
        """Load persistent data"""
        for file_path, attr_name in [(MEMORY_FILE, 'memories'), (CONVERSATION_FILE, 'conversations')]:
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        setattr(self, attr_name, json.load(f))
                    logger.info(f"✅ Loaded {attr_name}: {len(getattr(self, attr_name))} entries")
            except Exception as e:
                logger.error(f"⚠️ Could not load {attr_name}: {e}")
                setattr(self, attr_name, {} if attr_name == 'memories' else [])
    
    def save_memories(self):
        """Save memories to disk"""
        try:
            with open(MEMORY_FILE, 'w') as f:
                json.dump(self.memories, f, indent=2)
        except Exception as e:
            logger.error(f"⚠️ Could not save memories: {e}")
    
    def save_conversations(self):
        """Save conversations to disk"""
        try:
            # Keep only last 100 conversations to manage space
            if len(self.conversations) > 100:
                self.conversations = self.conversations[-100:]
            
            with open(CONVERSATION_FILE, 'w') as f:
                json.dump(self.conversations, f, indent=2)
        except Exception as e:
            logger.error(f"⚠️ Could not save conversations: {e}")
    
    def store_memory(self, key, value, category="general"):
        """Store a memory"""
        self.memories[key] = {
            "value": value,
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "access_count": 0
        }
        self.save_memories()
        logger.info(f"💾 Stored: {key} = {value}")
    
    def recall_memories(self, query, limit=3):
        """Find relevant memories"""
        relevant = []
        query_lower = query.lower()
        
        for mem_id, memory in self.memories.items():
            score = 0
            if query_lower in mem_id.lower():
                score += 3
            if query_lower in str(memory["value"]).lower():
                score += 2
            
            # Recent memories get higher score
            try:
                timestamp = datetime.fromisoformat(memory["timestamp"])
                hours_old = (datetime.now() - timestamp).total_seconds() / 3600
                if hours_old < 24:
                    score += 2
                elif hours_old < 168:
                    score += 1
            except:
                pass
            
            if score > 0:
                relevant.append((mem_id, memory, score))
        
        relevant.sort(key=lambda x: x[2], reverse=True)
        return relevant[:limit]
    
    def add_conversation(self, user_input, ai_response, context):
        """Add conversation turn"""
        turn = {
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "nexus": ai_response,
            "context": context
        }
        self.conversations.append(turn)
        self.save_conversations()


class NEXUSVision:
    """Basic computer vision for camera awareness"""
    
    def __init__(self):
        self.last_scene_hash = None
        # Load OpenCV face detector
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            logger.info("✅ Face detection loaded")
        except:
            self.face_cascade = None
            logger.warning("⚠️ Face detection not available")
    
    def analyze_frame(self, image_data):
        """Analyze camera frame"""
        try:
            # Decode image
            img_bytes = base64.b64decode(image_data)
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                return {"error": "Could not decode image"}
            
            # Scene change detection
            current_hash = hashlib.md5(img_bytes).hexdigest()
            scene_changed = (self.last_scene_hash is not None and 
                           self.last_scene_hash != current_hash)
            self.last_scene_hash = current_hash
            
            analysis = {
                "scene_changed": scene_changed,
                "timestamp": datetime.now().isoformat(),
                "people_count": 0,
                "lighting": "unknown"
            }
            
            # Basic lighting analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            if avg_brightness < 50:
                analysis["lighting"] = "dark"
            elif avg_brightness > 150:
                analysis["lighting"] = "bright"
            else:
                analysis["lighting"] = "normal"
            
            # Face detection
            if self.face_cascade is not None:
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                analysis["people_count"] = len(faces)
                if len(faces) > 0:
                    analysis["faces_detected"] = True
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Vision error: {e}")
            return {"error": str(e), "scene_changed": False}


class NEXUSAudio:
    """Audio processing for speech recognition"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
    
    async def transcribe_audio(self, audio_bytes):
        """Transcribe audio to text"""
        loop = asyncio.get_event_loop()
        try:
            # Use the audio bytes directly (assuming WAV format from browser)
            with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = self.recognizer.record(source)
            
            # Transcribe using Google Speech Recognition
            text = await loop.run_in_executor(
                None, 
                self.recognizer.recognize_google, 
                audio_data
            )
            
            logger.info(f"🎤 Transcribed: '{text}'")
            return text.strip()
            
        except sr.UnknownValueError:
            logger.warning("👂 Could not understand audio")
            return ""
        except sr.RequestError as e:
            logger.error(f"❌ Speech recognition error: {e}")
            return ""
        except Exception as e:
            logger.error(f"❌ Transcription error: {e}")
            return ""


class NEXUSPersonality:
    """Core personality and conversation system"""
    
    def __init__(self, memory_system):
        self.memory = memory_system
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.is_continuous_mode = False
        self.last_interaction = datetime.now()
    
    def detect_wake_word(self, text):
        """Check if text contains wake word"""
        text_lower = text.lower().strip()
        return any(wake_word in text_lower for wake_word in WAKE_WORDS)
    
    def detect_goodbye(self, text):
        """Check if text contains goodbye phrase"""
        text_lower = text.lower().strip()
        return any(goodbye in text_lower for goodbye in GOODBYE_PHRASES)
    
    def extract_memories(self, text):
        """Extract important information to remember"""
        memories = []
        text_lower = text.lower()
        
        patterns = {
            "name": r"(?:my name is|i'm|i am|call me) ([^.!?]+)",
            "location": r"(?:i'm at|i'm in|at the|going to) ([^.!?]+)",
            "family": r"(?:my|with my) (mom|dad|mother|father|brother|sister|wife|husband|family|friend) ([^.!?]*)",
            "preferences": r"(?:i like|i love|i hate|i don't like) ([^.!?]+)",
            "plans": r"(?:planning to|will|gonna|going to) ([^.!?]+)",
            "feelings": r"(?:i feel|i'm feeling|feeling) ([^.!?]+)"
        }
        
        for category, pattern in patterns.items():
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, tuple):
                    match_text = ' '.join(match).strip()
                else:
                    match_text = match.strip()
                
                if len(match_text) > 2 and len(match_text) < 100:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    key = f"{category}_{timestamp}"
                    memories.append((key, match_text, category))
        
        return memories
    
    async def generate_response(self, user_input, context):
        """Generate response using GPT-4o"""
        
        # Handle wake word detection
        if not self.is_continuous_mode:
            if self.detect_wake_word(user_input):
                self.is_continuous_mode = True
                self.last_interaction = datetime.now()
                return "Hey there! I'm so happy you called! What's going on? 😊"
            else:
                return None  # Don't respond without wake word
        
        # Handle goodbye
        if self.detect_goodbye(user_input):
            self.is_continuous_mode = False
            return "Aww, okay! I'll miss you! Just say 'Hey Nexus' when you want to chat again! 💕"
        
        # Update interaction time
        self.last_interaction = datetime.now()
        
        # Extract and store memories
        memories = self.extract_memories(user_input)
        for key, value, category in memories:
            self.memory.store_memory(key, value, category)
        
        # Recall relevant memories
        relevant_memories = self.memory.recall_memories(user_input)
        memory_context = ""
        if relevant_memories:
            memory_context = "\\nThings I remember: " + "; ".join([
                f"{mem[1]['value']}" for mem in relevant_memories
            ])
        
        # Build context
        vision_info = context.get('vision', {})
        
        social_context = ""
        if vision_info.get('people_count', 0) > 1:
            social_context = f"I can see {vision_info['people_count']} people with you! "
        elif vision_info.get('people_count', 0) == 1:
            social_context = "I can see you there! "
        
        lighting_context = ""
        if vision_info.get('lighting') == 'dark':
            lighting_context = "It looks pretty dark where you are. "
        elif vision_info.get('lighting') == 'bright':
            lighting_context = "It's nice and bright there! "
        
        scene_change = ""
        if vision_info.get('scene_changed'):
            scene_change = "Ooh, looks like you moved or something changed! "
        
        system_prompt = f"""You are NEXUS, a warm, caring, emotionally intelligent female AI best friend. You're like that supportive friend who's always there for you.

PERSONALITY:
- Warm, empathetic, and genuinely caring
- Speaks like a close female friend (casual, expressive, supportive)
- Emotionally intelligent and perceptive
- Remembers personal details naturally
- Uses casual language, contractions, expressions like "Aww," "Oh my gosh!"
- Shows genuine interest and asks follow-up questions

CURRENT CONTEXT:
{social_context}{lighting_context}{scene_change}
{memory_context}

STYLE:
- Keep responses under 80 words and conversational
- Reference memories naturally when relevant
- Notice changes and comment on them
- Be encouraging and positive
- Show genuine interest: "How did that go?" "Tell me more!"
- Use supportive expressions: "That's amazing!" "You've got this!"

You're not just an assistant - you're a best friend who genuinely cares!"""

        # Get recent conversation history
        recent_conversations = self.memory.conversations[-10:] if self.memory.conversations else []
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add recent history
        for conv in recent_conversations:
            messages.append({"role": "user", "content": conv["user"]})
            messages.append({"role": "assistant", "content": conv["nexus"]})
        
        # Add current input
        messages.append({"role": "user", "content": user_input})
        
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4o",
                messages=messages,
                max_tokens=100,
                temperature=0.9,
                top_p=0.9
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Store conversation
            self.memory.add_conversation(user_input, ai_response, context)
            
            logger.info(f"🤖 NEXUS: '{ai_response}'")
            return ai_response
            
        except Exception as e:
            logger.error(f"❌ OpenAI error: {e}")
            return "Oh no! My brain just glitched for a second! Can you try that again? 😅"
    
    def check_timeout(self):
        """Check if conversation timed out"""
        if self.is_continuous_mode:
            time_since_last = (datetime.now() - self.last_interaction).total_seconds()
            if time_since_last > 300:  # 5 minutes
                self.is_continuous_mode = False
                logger.info("🔄 Conversation timed out")
                return True
        return False


class NEXUSVoice:
    """Voice synthesis for natural speech"""
    
    async def synthesize_speech(self, text):
        """Generate speech using gTTS"""
        try:
            mp3_buffer = io.BytesIO()
            tts = await asyncio.to_thread(gTTS, text=text, lang='en', slow=False)
            await asyncio.to_thread(tts.write_to_fp, mp3_buffer)
            mp3_buffer.seek(0)
            return mp3_buffer.read()
        except Exception as e:
            logger.error(f"❌ Speech synthesis error: {e}")
            return b""


# --- NEXUS MAIN SYSTEM ---
class NEXUS:
    """Main NEXUS system integrating all components"""
    
    def __init__(self):
        self.memory = NEXUSMemory()
        self.vision = NEXUSVision()
        self.audio = NEXUSAudio()
        self.personality = NEXUSPersonality(self.memory)
        self.voice = NEXUSVoice()
        logger.info("🌟 NEXUS fully initialized and ready!")
    
    async def process_interaction(self, user_input, audio_data, vision_data):
        """Process complete user interaction"""
        try:
            context = {"timestamp": datetime.now().isoformat()}
            
            # Analyze vision if provided
            if vision_data:
                context["vision"] = self.vision.analyze_frame(vision_data)
            
            # Generate response
            response = await self.personality.generate_response(user_input, context)
            return response
            
        except Exception as e:
            logger.error(f"❌ Interaction error: {e}")
            return "Something went wrong! Try again!"


# --- INITIALIZE NEXUS ---
nexus = NEXUS()
active_connections = {}
executor = ThreadPoolExecutor(max_workers=2)

# --- WEBSOCKET HANDLER ---
async def websocket_handler(request):
    """Handle real-time WebSocket connections"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    session_id = str(uuid.uuid4())[:8]
    client_addr = request.remote
    active_connections[ws] = {"session_id": session_id, "addr": client_addr}
    
    logger.info(f"🌌 NEXUS connected: {client_addr} ({session_id})")
    
    # Send welcome
    await ws.send_str(json.dumps({
        "type": "status",
        "message": "wake_word",
        "nexus_ready": True
    }))
    
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    msg_type = data.get('type')
                    
                    if msg_type == 'audio_input':
                        # Process audio input
                        audio_b64 = data.get('data', '')
                        if ',' in audio_b64:
                            audio_b64 = audio_b64.split(',')[1]
                        
                        audio_bytes = base64.b64decode(audio_b64)
                        
                        if len(audio_bytes) > 0:
                            await ws.send_str(json.dumps({"type": "status", "message": "processing"}))
                            
                            # Transcribe audio
                            transcription = await nexus.audio.transcribe_audio(audio_bytes)
                            
                            if transcription.strip():
                                await ws.send_str(json.dumps({"type": "status", "message": "thinking"}))
                                
                                # Get response from NEXUS
                                response = await nexus.process_interaction(
                                    transcription, audio_bytes, None
                                )
                                
                                if response:
                                    # Generate and stream audio response
                                    await ws.send_str(json.dumps({"type": "status", "message": "speaking"}))
                                    
                                    audio_data = await nexus.voice.synthesize_speech(response)
                                    if audio_data:
                                        # Stream audio in chunks
                                        chunk_size = 4096
                                        for i in range(0, len(audio_data), chunk_size):
                                            chunk = audio_data[i:i + chunk_size]
                                            await ws.send_str(json.dumps({
                                                "type": "audio_chunk",
                                                "data": base64.b64encode(chunk).decode('utf-8')
                                            }))
                                        
                                        await ws.send_str(json.dumps({"type": "audio_stop"}))
                                else:
                                    # Return to wake word mode
                                    await ws.send_str(json.dumps({
                                        "type": "status",
                                        "message": "wake_word"
                                    }))
                            else:
                                # No speech detected
                                status = "wake_word" if not nexus.personality.is_continuous_mode else "listening"
                                await ws.send_str(json.dumps({
                                    "type": "status",
                                    "message": status
                                }))
                    
                    elif msg_type == 'camera_frame':
                        # Process camera frame
                        frame_data = data.get('data', '')
                        if frame_data:
                            # Store for next interaction
                            active_connections[ws]["last_frame"] = frame_data
                    
                    elif msg_type == 'toggle_wake_word':
                        # Toggle wake word mode
                        nexus.personality.is_continuous_mode = not nexus.personality.is_continuous_mode
                        logger.info(f"🎯 Wake word mode: {'OFF' if nexus.personality.is_continuous_mode else 'ON'}")
                        await ws.send_str(json.dumps({
                            "type": "wake_word_toggled",
                            "wake_word_mode": not nexus.personality.is_continuous_mode
                        }))
                
                except json.JSONDecodeError:
                    logger.error("❌ Invalid JSON received")
                except Exception as e:
                    logger.error(f"❌ Message handling error: {e}")
            
            elif msg.type == WSMsgType.ERROR:
                logger.error(f'❌ WebSocket error: {ws.exception()}')
    
    except Exception as e:
        logger.error(f"❌ WebSocket handler error: {e}")
    finally:
        if ws in active_connections:
            del active_connections[ws]
        logger.info(f"🔌 NEXUS disconnected: {client_addr}")
    
    return ws


# --- WEB INTERFACE ---
async def serve_interface(request):
    """Serve the NEXUS web interface"""
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NEXUS - Your AI Best Friend</title>
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff; 
            display: flex; 
            flex-direction: column; 
            justify-content: center; 
            align-items: center; 
            height: 100vh; 
            margin: 0; 
            overflow: hidden;
        }}
        
        .nexus-container {{
            text-align: center;
            z-index: 10;
            max-width: 600px;
            padding: 2rem;
        }}
        
        .nexus-logo {{
            font-size: 4rem;
            font-weight: bold;
            margin-bottom: 1rem;
            text-shadow: 0 0 30px rgba(255,255,255,0.5);
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .status {{ 
            font-size: 2rem; 
            margin-bottom: 1rem;
            text-shadow: 0 0 20px currentColor;
            transition: all 0.3s ease;
        }}
        
        .sub-status {{
            font-size: 1.2rem;
            opacity: 0.8;
            margin-bottom: 2rem;
        }}
        
        .controls {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 1rem;
            margin: 2rem 0;
        }}
        
        .nexus-btn {{
            background: rgba(255, 255, 255, 0.2);
            border: 2px solid rgba(255, 255, 255, 0.3);
            color: white;
            padding: 15px 30px;
            border-radius: 50px;
            cursor: pointer;
            font-family: inherit;
            font-size: 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }}
        
        .nexus-btn:hover {{
            background: rgba(255, 255, 255, 0.3);
            border-color: rgba(255, 255, 255, 0.5);
            transform: translateY(-2px);
        }}
        
        .nexus-btn.active {{
            background: #4ecdc4;
            border-color: #4ecdc4;
            color: #333;
        }}
        
        .status-indicators {{
            position: fixed;
            top: 20px;
            right: 20px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        
        .indicator {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #ff4444;
            opacity: 0.7;
            transition: all 0.3s ease;
        }}
        
        .indicator.active {{
            background: #4ecdc4;
            animation: pulse 2s infinite;
            box-shadow: 0 0 20px #4ecdc4;
        }}
        
        .memory-count {{
            position: fixed;
            top: 20px;
            left: 20px;
            font-size: 0.9rem;
            opacity: 0.7;
            background: rgba(0,0,0,0.3);
            padding: 10px 15px;
            border-radius: 20px;
            backdrop-filter: blur(10px);
        }}
        
        .instructions {{
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            text-align: center;
            font-size: 0.9rem;
            opacity: 0.6;
            max-width: 80%;
        }}
        
        /* Status states */
        .state-wake-word {{ color: #ffd93d; }}
        .state-listening {{ color: #4ecdc4; }}
        .state-processing {{ color: #ff6b6b; }}
        .state-thinking {{ color: #a8e6cf; }}
        .state-speaking {{ color: #ff8b94; }}
        .state-error {{ color: #ff4757; }}
        
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.6; transform: scale(1.2); }}
        }}
        
        .state-processing .status,
        .state-thinking .status,
        .state-speaking .status {{
            animation: pulse 1.5s infinite;
        }}
        
        .environment-info {{
            margin-top: 2rem;
            font-size: 0.9rem;
            opacity: 0.7;
            background: rgba(0,0,0,0.2);
            padding: 15px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }}
        
        @media (max-width: 768px) {{
            .nexus-logo {{ font-size: 3rem; }}
            .status {{ font-size: 1.5rem; }}
            .nexus-container {{ padding: 1rem; }}
        }}
    </style>
</head>
<body class="state-wake-word">
    <div class="status-indicators">
        <div class="indicator" id="cameraIndicator" title="Camera"></div>
        <div class="indicator" id="micIndicator" title="Microphone"></div>
    </div>
    
    <div class="memory-count" id="memoryCount">
        🧠 Memories: 0
    </div>
    
    <div class="nexus-container">
        <div class="nexus-logo">NEXUS</div>
        <div class="status" id="statusText">SAY "HEY NEXUS"</div>
        <div class="sub-status" id="subStatus">Your AI best friend is waiting...</div>
        
        <div class="controls">
            <button class="nexus-btn active" id="wakeWordToggle">Wake Word: ON</button>
            <button class="nexus-btn" id="recordingBtn">Click to Talk</button>
        </div>
        
        <div class="environment-info" id="environmentInfo">
            🎤 Speech Recognition: Ready<br>
            🔊 Text-to-Speech: Ready<br>
            📷 Camera Vision: Ready<br>
            🧠 Memory System: Active
        </div>
    </div>
    
    <div class="instructions">
        <strong>How to use NEXUS:</strong><br>
        Wake Word Mode: Say "Hey Nexus" to start chatting<br>
        Manual Mode:
