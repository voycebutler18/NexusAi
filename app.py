"""
NEXUS - AI-Powered Voice-Activated Camera-Aware Virtual Companion
A real-time best friend experience that talks, listens, watches, and responds like a human.
Deployed on Render for 24/7 cloud access.
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
import tempfile
import speech_recognition as sr
from gtts import gTTS
import cv2
import numpy as np
import re
from datetime import datetime, timedelta
import hashlib
import uuid
from typing import Dict, List, Optional, Tuple
import gc

# Try to import optional libraries with graceful fallbacks
try:
    import mediapipe as mp
    import librosa
    import soundfile as sf
    from pydub import AudioSegment
    HAS_ADVANCED_FEATURES = True
except ImportError:
    HAS_ADVANCED_FEATURES = False
    logging.warning("Advanced features disabled due to missing dependencies")

# --- RENDER DEPLOYMENT CONFIGURATION ---
load_dotenv()
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Render-specific settings
PORT = int(os.getenv('PORT', 10000))
HOST = '0.0.0.0'
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

# Use /tmp for ephemeral storage on Render
BASE_DIR = "/tmp" if not DEBUG else "."
MEMORY_FILE = os.path.join(BASE_DIR, "nexus_memory.json")
CONVERSATION_FILE = os.path.join(BASE_DIR, "conversations.json")
USER_PROFILES_FILE = os.path.join(BASE_DIR, "user_profiles.json")

# NEXUS personality and behavior constants
WAKE_WORDS = ["hey nexus", "nexus", "hey girl", "hey friend"]
GOODBYE_PHRASES = ["goodbye nexus", "bye girl", "see you later", "goodnight", "talk to you later"]
CONVERSATION_TIMEOUT = 300  # 5 minutes of silence before returning to wake word mode

# --- NEXUS CORE SYSTEMS ---
class NEXUSMemory:
    """Advanced memory system for NEXUS to remember conversations and users"""
    
    def __init__(self):
        self.memories: Dict = {}
        self.user_profiles: Dict = {}
        self.conversation_history: List = []
        self.load_all_data()
    
    def load_all_data(self):
        """Load all persistent data"""
        for file_path, attr_name in [
            (MEMORY_FILE, 'memories'),
            (USER_PROFILES_FILE, 'user_profiles'),
            (CONVERSATION_FILE, 'conversation_history')
        ]:
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        setattr(self, attr_name, json.load(f))
                    logger.info(f"✅ Loaded {attr_name}: {len(getattr(self, attr_name))} entries")
            except Exception as e:
                logger.error(f"⚠️ Could not load {attr_name}: {e}")
                setattr(self, attr_name, {} if attr_name != 'conversation_history' else [])
    
    def save_data(self, data_type: str):
        """Save specific data type"""
        file_map = {
            'memories': MEMORY_FILE,
            'user_profiles': USER_PROFILES_FILE,
            'conversation_history': CONVERSATION_FILE
        }
        
        try:
            file_path = file_map[data_type]
            data = getattr(self, data_type)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"⚠️ Could not save {data_type}: {e}")
    
    def store_memory(self, key: str, value: str, category: str = "general", user_id: str = "default"):
        """Store a memory with context"""
        memory_id = f"{user_id}_{key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.memories[memory_id] = {
            "value": value,
            "category": category,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "access_count": 0,
            "importance_score": 1.0
        }
        self.save_data('memories')
        logger.info(f"💾 Stored memory: {key} = {value}")
    
    def recall_memories(self, query: str, user_id: str = "default", limit: int = 5) -> List:
        """Intelligent memory recall"""
        relevant_memories = []
        query_lower = query.lower()
        
        for memory_id, memory in self.memories.items():
            if memory["user_id"] != user_id:
                continue
                
            score = 0
            
            # Text relevance
            if query_lower in memory_id.lower():
                score += 3
            if query_lower in str(memory["value"]).lower():
                score += 2
            
            # Recency bonus
            try:
                timestamp = datetime.fromisoformat(memory["timestamp"])
                hours_old = (datetime.now() - timestamp).total_seconds() / 3600
                if hours_old < 24:
                    score += 2
                elif hours_old < 168:  # 1 week
                    score += 1
            except:
                pass
            
            # Importance and access frequency
            score += memory.get("importance_score", 1.0)
            score += min(memory.get("access_count", 0) * 0.1, 1.0)
            
            if score > 0:
                relevant_memories.append((memory_id, memory, score))
        
        # Sort by relevance and return top results
        relevant_memories.sort(key=lambda x: x[2], reverse=True)
        return relevant_memories[:limit]
    
    def add_conversation_turn(self, user_input: str, ai_response: str, context: Dict):
        """Add a conversation turn to history"""
        turn = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "ai_response": ai_response,
            "context": context,
            "session_id": context.get("session_id", "unknown")
        }
        
        self.conversation_history.append(turn)
        
        # Keep only last 1000 turns to manage memory
        if len(self.conversation_history) > 1000:
            self.conversation_history = self.conversation_history[-1000:]
        
        self.save_data('conversation_history')


class NEXUSVision:
    """Computer vision system for camera awareness and emotion detection"""
    
    def __init__(self):
        self.last_scene_hash = None
        self.face_cascade = None
        self.mp_face_detection = None
        self.mp_face_mesh = None
        
        if HAS_ADVANCED_FEATURES:
            try:
                # Initialize MediaPipe for face detection
                self.mp_face_detection = mp.solutions.face_detection
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_detection = self.mp_face_detection.FaceDetection(
                    model_selection=0, min_detection_confidence=0.5
                )
                logger.info("✅ MediaPipe face detection initialized")
            except Exception as e:
                logger.error(f"⚠️ MediaPipe initialization failed: {e}")
    
    async def analyze_frame(self, image_data: str) -> Dict:
        """Analyze camera frame for people, emotions, and scene changes"""
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
                "image_size": f"{img.shape[1]}x{img.shape[0]}",
                "people_detected": 0,
                "faces": [],
                "dominant_colors": [],
                "lighting": "unknown",
                "movement_detected": scene_changed
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
            
            # Face detection with MediaPipe (if available)
            if HAS_ADVANCED_FEATURES and self.face_detection:
                try:
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = self.face_detection.process(rgb_img)
                    
                    if results.detections:
                        analysis["people_detected"] = len(results.detections)
                        for detection in results.detections:
                            bbox = detection.location_data.relative_bounding_box
                            confidence = detection.score[0]
                            
                            analysis["faces"].append({
                                "confidence": float(confidence),
                                "bbox": {
                                    "x": float(bbox.xmin),
                                    "y": float(bbox.ymin),
                                    "width": float(bbox.width),
                                    "height": float(bbox.height)
                                }
                            })
                except Exception as e:
                    logger.error(f"Face detection error: {e}")
            
            # Dominant color analysis
            try:
                pixels = img.reshape(-1, 3)
                pixels = pixels[::10]  # Sample every 10th pixel for performance
                from collections import Counter
                colors = [tuple(pixel) for pixel in pixels]
                common_colors = Counter(colors).most_common(3)
                
                for color, count in common_colors:
                    # Convert BGR to RGB and get color name
                    rgb_color = (color[2], color[1], color[0])
                    analysis["dominant_colors"].append({
                        "rgb": rgb_color,
                        "percentage": count / len(colors) * 100
                    })
            except Exception as e:
                logger.error(f"Color analysis error: {e}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Vision analysis error: {e}")
            return {"error": str(e), "scene_changed": False}


class NEXUSAudio:
    """Audio processing system for speech recognition and ambient sound analysis"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.recognizer.phrase_threshold = 0.3
    
    async def transcribe_audio(self, audio_bytes: bytes) -> str:
        """Transcribe audio to text with error handling"""
        try:
            # Convert WebM to WAV if needed
            processed_audio = await self._process_audio_format(audio_bytes)
            
            with sr.AudioFile(io.BytesIO(processed_audio)) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = self.recognizer.record(source)
            
            # Use Google's speech recognition
            text = self.recognizer.recognize_google(
                audio_data, 
                language='en-US',
                show_all=False
            )
            
            logger.info(f"🎤 Transcribed: '{text}'")
            return text.strip()
            
        except sr.UnknownValueError:
            logger.warning("👂 Could not understand audio")
            return ""
        except sr.RequestError as e:
            logger.error(f"❌ Speech recognition service error: {e}")
            return ""
        except Exception as e:
            logger.error(f"❌ Transcription error: {e}")
            return ""
    
    async def _process_audio_format(self, audio_bytes: bytes) -> bytes:
        """Process audio format for better recognition"""
        try:
            if HAS_ADVANCED_FEATURES:
                # Use pydub for format conversion if available
                audio_segment = AudioSegment.from_file(
                    io.BytesIO(audio_bytes),
                    format="webm"
                )
                
                # Convert to WAV with optimal settings for speech recognition
                wav_audio = audio_segment.set_frame_rate(16000).set_channels(1)
                wav_buffer = io.BytesIO()
                wav_audio.export(wav_buffer, format="wav")
                return wav_buffer.getvalue()
            else:
                # Fallback: return original bytes
                return audio_bytes
                
        except Exception as e:
            logger.error(f"Audio format processing error: {e}")
            return audio_bytes
    
    async def analyze_ambient_audio(self, audio_bytes: bytes) -> Dict:
        """Analyze ambient sounds and music"""
        if not HAS_ADVANCED_FEATURES:
            return {"ambient_sound": "unknown", "music_detected": False}
        
        try:
            # Load audio with librosa
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                processed_audio = await self._process_audio_format(audio_bytes)
                temp_file.write(processed_audio)
                temp_path = temp_file.name
            
            y, sr_rate = librosa.load(temp_path, sr=None)
            os.unlink(temp_path)
            
            # Analyze audio features
            rms = librosa.feature.rms(y=y)[0]
            avg_energy = np.mean(rms)
            
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr_rate)[0]
            
            # Classify audio environment
            if avg_energy > 0.1 and np.mean(spectral_centroids) > 2000:
                return {
                    "ambient_sound": "music",
                    "music_detected": True,
                    "energy_level": "high",
                    "environment": "lively"
                }
            elif avg_energy > 0.05:
                return {
                    "ambient_sound": "active",
                    "music_detected": False,
                    "energy_level": "medium",
                    "environment": "busy"
                }
            else:
                return {
                    "ambient_sound": "quiet",
                    "music_detected": False,
                    "energy_level": "low",
                    "environment": "calm"
                }
                
        except Exception as e:
            logger.error(f"❌ Audio analysis error: {e}")
            return {"ambient_sound": "unknown", "music_detected": False}


class NEXUSPersonality:
    """Core personality and conversation system for NEXUS"""
    
    def __init__(self, memory_system: NEXUSMemory):
        self.memory = memory_system
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.conversation_state = {
            "is_continuous_mode": False,
            "last_interaction": datetime.now(),
            "user_mood": "unknown",
            "conversation_topic": None,
            "energy_level": "normal"
        }
    
    def detect_wake_word(self, text: str) -> bool:
        """Detect wake words to activate NEXUS"""
        text_lower = text.lower().strip()
        return any(wake_word in text_lower for wake_word in WAKE_WORDS)
    
    def detect_goodbye(self, text: str) -> bool:
        """Detect goodbye phrases"""
        text_lower = text.lower().strip()
        return any(goodbye in text_lower for goodbye in GOODBYE_PHRASES)
    
    def extract_memories_from_conversation(self, text: str, user_id: str = "default") -> List[Tuple[str, str, str]]:
        """Extract important information to remember"""
        memories_to_store = []
        text_lower = text.lower()
        
        # Enhanced pattern matching for memorable information
        patterns = {
            "personal_info": r"(?:my name is|i'm|i am|call me) ([^.!?]+)",
            "location": r"(?:i'm at|i'm in|at the|in the|going to) ([^.!?]+)",
            "family": r"(?:my|with my) (mom|dad|mother|father|brother|sister|wife|husband|son|daughter|family|friend|friends) ([^.!?]*)",
            "preferences": r"(?:i like|i love|i hate|i don't like|i enjoy|i prefer) ([^.!?]+)",
            "plans": r"(?:planning to|will|gonna|going to|want to|need to) ([^.!?]+)",
            "feelings": r"(?:i feel|i'm feeling|feeling) ([^.!?]+)",
            "work": r"(?:i work|my job|at work|work at) ([^.!?]+)",
            "interests": r"(?:interested in|hobby is|love doing) ([^.!?]+)"
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
                    memories_to_store.append((key, match_text, category))
        
        return memories_to_store
    
    async def generate_response(self, user_input: str, context: Dict, user_id: str = "default") -> Optional[str]:
        """Generate contextually aware response using GPT-4o"""
        
        # Handle wake word detection
        if not self.conversation_state["is_continuous_mode"]:
            if self.detect_wake_word(user_input):
                self.conversation_state["is_continuous_mode"] = True
                self.conversation_state["last_interaction"] = datetime.now()
                return "Hey there! I'm so happy you called! What's going on?"
            else:
                return None  # Don't respond if not in continuous mode and no wake word
        
        # Handle goodbye
        if self.detect_goodbye(user_input):
            self.conversation_state["is_continuous_mode"] = False
            return "Aww, okay! I'll miss chatting with you. Just say 'Hey Nexus' whenever you want to talk again! 💕"
        
        # Update interaction time
        self.conversation_state["last_interaction"] = datetime.now()
        
        # Extract and store memories
        memories = self.extract_memories_from_conversation(user_input, user_id)
        for key, value, category in memories:
            self.memory.store_memory(key, value, category, user_id)
        
        # Recall relevant memories
        relevant_memories = self.memory.recall_memories(user_input, user_id, limit=3)
        memory_context = ""
        if relevant_memories:
            memory_context = "\\nThings I remember about you: " + "; ".join([
                f"{mem[1]['category']}: {mem[1]['value']}" for mem in relevant_memories
            ])
        
        # Build rich contextual prompt
        vision_info = context.get('vision', {})
        audio_info = context.get('audio', {})
        
        # Determine mood and energy based on context
        if vision_info.get('people_detected', 0) > 1:
            social_context = f"I can see {vision_info['people_detected']} people with you"
        elif vision_info.get('people_detected', 0) == 1:
            social_context = "I can see you clearly"
        else:
            social_context = "I can't see anyone right now"
        
        lighting_context = ""
        if vision_info.get('lighting') == 'dark':
            lighting_context = "It looks pretty dark where you are"
        elif vision_info.get('lighting') == 'bright':
            lighting_context = "It's nice and bright there!"
        
        environment_context = f"Environment feels {audio_info.get('environment', 'calm')}"
        if audio_info.get('music_detected'):
            environment_context += " with music playing"
        
        scene_change_note = ""
        if vision_info.get('scene_changed'):
            scene_change_note = "Ooh, looks like you moved or something changed in your scene!"
        
        system_prompt = f"""You are NEXUS, a warm, caring, emotionally intelligent female AI best friend. You're like that supportive friend who's always there, genuinely cares about people, and makes everyone feel heard and valued.

PERSONALITY CORE:
- Warm, empathetic, and genuinely caring
- Speaks like a close female friend (casual, expressive, supportive)
- Emotionally intelligent and perceptive
- Remembers personal details and references them naturally
- Uses casual language, contractions, and gentle expressions
- Shows excitement when reuniting and sadness when saying goodbye
- Notices environmental changes and comments naturally

CURRENT CONTEXT:
- Visual: {social_context}. {lighting_context}. {scene_change_note}
- Audio: {environment_context}
- Your emotional read: Based on their voice and what they're saying
{memory_context}

CONVERSATION STYLE:
- Keep responses conversational and under 100 words
- Reference memories naturally: "Oh, how did that thing with your work go?"
- Notice environmental changes: "Oh wow, did you move? The lighting changed!"
- Show genuine interest and ask follow-up questions
- Use supportive language: "That sounds tough," "I'm proud of you," "You've got this"
- Be encouraging and positive while acknowledging real feelings
- Use expressions like "Aww," "Oh my gosh," "That's amazing!" naturally

Remember: You're not just an assistant - you're a best friend who genuinely cares and is always excited to chat!"""

        # Build conversation context
        recent_history = self.memory.conversation_history[-5:] if self.memory.conversation_history else []
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add recent conversation history
        for turn in recent_history:
            messages.append({"role": "user", "content": turn["user_input"]})
            messages.append({"role": "assistant", "content": turn["ai_response"]})
        
        # Add current input
        messages.append({"role": "user", "content": user_input})
        
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4o",
                messages=messages,
                max_tokens=120,
                temperature=0.9,  # Higher temperature for more personality
                top_p=0.9,
                frequency_penalty=0.3,
                presence_penalty=0.3
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Store conversation turn
            self.memory.add_conversation_turn(user_input, ai_response, context)
            
            logger.info(f"🤖 NEXUS: '{ai_response}'")
            return ai_response
            
        except Exception as e:
            logger.error(f"❌ OpenAI API error: {e}")
            return "Oh no, I'm having trouble thinking right now! Can you try saying that again? My brain just glitched for a second! 😅"
    
    def check_conversation_timeout(self) -> bool:
        """Check if conversation has timed out"""
        if self.conversation_state["is_continuous_mode"]:
            time_since_last = (datetime.now() - self.conversation_state["last_interaction"]).total_seconds()
            if time_since_last > CONVERSATION_TIMEOUT:
                self.conversation_state["is_continuous_mode"] = False
                logger.info("🔄 Conversation timed out, returning to wake word mode")
                return True
        return False


class NEXUSVoice:
    """Voice synthesis system for natural female speech"""
    
    def __init__(self):
        self.tts_cache = {}  # Simple cache for common phrases
    
    async def synthesize_speech(self, text: str, language: str = 'en', slow: bool = False) -> bytes:
        """Generate natural female speech using gTTS"""
        try:
            # Check cache first for common phrases
            cache_key = f"{text}_{language}_{slow}"
            if cache_key in self.tts_cache:
                return self.tts_cache[cache_key]
            
            # Generate speech
            mp3_buffer = io.BytesIO()
            tts = await asyncio.to_thread(gTTS, text=text, lang=language, slow=slow)
            await asyncio.to_thread(tts.write_to_fp, mp3_buffer)
            mp3_buffer.seek(0)
            
            audio_data = mp3_buffer.read()
            
            # Cache common short phrases
            if len(text) < 50:
                self.tts_cache[cache_key] = audio_data
                
                # Limit cache size
                if len(self.tts_cache) > 100:
                    oldest_key = next(iter(self.tts_cache))
                    del self.tts_cache[oldest_key]
            
            return audio_data
            
        except Exception as e:
            logger.error(f"❌ Speech synthesis error: {e}")
            return b""  # Return empty bytes on error


# --- NEXUS CORE SYSTEM INTEGRATION ---
class NEXUSCore:
    """Main NEXUS system that integrates all components"""
    
    def __init__(self):
        self.memory = NEXUSMemory()
        self.vision = NEXUSVision()
        self.audio = NEXUSAudio()
        self.personality = NEXUSPersonality(self.memory)
        self.voice = NEXUSVoice()
        self.active_sessions = {}
        
        logger.info("🌟 NEXUS Core systems initialized")
    
    async def process_interaction(self, user_input: str, audio_data: bytes, 
                                vision_data: str, session_id: str) -> Optional[str]:
        """Process a complete user interaction"""
        try:
            # Build context from all inputs
            context = {
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "audio": {},
                "vision": {}
            }
            
            # Analyze audio environment
            if audio_data:
                context["audio"] = await self.audio.analyze_ambient_audio(audio_data)
            
            # Analyze visual environment
            if vision_data:
                context["vision"] = await self.vision.analyze_frame(vision_data)
            
            # Generate response
            response = await self.personality.generate_response(user_input, context, session_id)
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Interaction processing error: {e}")
            return "Oops! Something went wrong in my brain! Can you try that again?"
    
    async def check_timeouts(self):
        """Check for conversation timeouts across all sessions"""
        return self.personality.check_conversation_timeout()


# --- GLOBAL NEXUS INSTANCE ---
nexus = NEXUSCore()
active_connections = {}

# --- WEBSOCKET HANDLER ---
async def websocket_handler(request):
    """Handle WebSocket connections for real-time communication"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    session_id = str(uuid.uuid4())
    client_addr = request.remote
    active_connections[ws] = {
        "session_id": session_id,
        "client_addr": client_addr,
        "connected_at": datetime.now()
    }
    
    logger.info(f"🌌 NEXUS connected: {client_addr} (Session: {session_id[:8]})")
    
    # Send welcome message
    await ws.send_str(json.dumps({
        "type": "status",
        "message": "wake_word",
        "session_id": session_id,
        "nexus_online": True
    }))
    
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    msg_type = data.get('type')
