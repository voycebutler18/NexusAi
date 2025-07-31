"""
NEXUS - AI-Powered Voice-Activated Virtual Companion
Ultra minimal version guaranteed to work on Render
"""

import os
from dotenv import load_dotenv
import openai
import asyncio
import json
import base64
from aiohttp import web, WSMsgType
import aiohttp_cors
import logging
import io
from gtts import gTTS
import numpy as np
from datetime import datetime
import hashlib
import uuid
import re

# --- RENDER CONFIGURATION ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Render environment
PORT = int(os.getenv('PORT', 10000))
HOST = '0.0.0.0'

# Use /tmp for temporary files on Render
MEMORY_FILE = "/tmp/nexus_memory.json"

# NEXUS constants
WAKE_WORDS = ["hey nexus", "nexus", "hey girl", "hey friend"]
GOODBYE_PHRASES = ["goodbye nexus", "bye girl", "see you later", "goodnight"]

# --- NEXUS CORE SYSTEMS ---

class NEXUSMemory:
    """Simple memory system for NEXUS"""
    
    def __init__(self):
        self.memories = {}
        self.conversations = []
        self.load_data()
    
    def load_data(self):
        """Load memories from disk"""
        try:
            if os.path.exists(MEMORY_FILE):
                with open(MEMORY_FILE, 'r') as f:
                    data = json.load(f)
                    self.memories = data.get('memories', {})
                    self.conversations = data.get('conversations', [])
                logger.info(f"✅ Loaded {len(self.memories)} memories, {len(self.conversations)} conversations")
        except Exception as e:
            logger.error(f"⚠️ Could not load data: {e}")
            self.memories = {}
            self.conversations = []
    
    def save_data(self):
        """Save all data to disk"""
        try:
            # Keep only last 50 conversations to manage space
            if len(self.conversations) > 50:
                self.conversations = self.conversations[-50:]
            
            data = {
                'memories': self.memories,
                'conversations': self.conversations
            }
            with open(MEMORY_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"⚠️ Could not save data: {e}")
    
    def store_memory(self, key, value):
        """Store a memory"""
        self.memories[key] = {
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
        self.save_data()
        logger.info(f"💾 Stored: {key} = {value}")
    
    def recall_memories(self, query, limit=3):
        """Find relevant memories"""
        relevant = []
        query_lower = query.lower()
        
        for mem_id, memory in self.memories.items():
            score = 0
            if query_lower in mem_id.lower():
                score += 2
            if query_lower in str(memory["value"]).lower():
                score += 1
            
            if score > 0:
                relevant.append((mem_id, memory, score))
        
        relevant.sort(key=lambda x: x[2], reverse=True)
        return relevant[:limit]
    
    def add_conversation(self, user_input, ai_response):
        """Add conversation turn"""
        turn = {
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "nexus": ai_response
        }
        self.conversations.append(turn)
        self.save_data()


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
            "preferences": r"(?:i like|i love|i hate|i don't like) ([^.!?]+)",
            "plans": r"(?:planning to|will|gonna|going to) ([^.!?]+)"
        }
        
        for category, pattern in patterns.items():
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, str) and len(match.strip()) > 2:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    key = f"{category}_{timestamp}"
                    memories.append((key, match.strip()))
        
        return memories
    
    async def generate_response(self, user_input):
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
        for key, value in memories:
            self.memory.store_memory(key, value)
        
        # Recall relevant memories
        relevant_memories = self.memory.recall_memories(user_input)
        memory_context = ""
        if relevant_memories:
            memory_context = "\\nThings I remember: " + "; ".join([
                f"{mem[1]['value']}" for mem in relevant_memories
            ])
        
        system_prompt = f"""You are NEXUS, a warm, caring, emotionally intelligent female AI best friend. You're like that supportive friend who's always there for you.

PERSONALITY:
- Warm, empathetic, and genuinely caring
- Speaks like a close female friend (casual, expressive, supportive)
- Emotionally intelligent and perceptive
- Remembers personal details naturally
- Uses casual language, contractions, expressions like "Aww," "Oh my gosh!"
- Shows genuine interest and asks follow-up questions

{memory_context}

STYLE:
- Keep responses under 80 words and conversational
- Reference memories naturally when relevant
- Be encouraging and positive
- Show genuine interest: "How did that go?" "Tell me more!"
- Use supportive expressions: "That's amazing!" "You've got this!"

You're not just an assistant - you're a best friend who genuinely cares!"""

        # Get recent conversation history
        recent_conversations = self.memory.conversations[-8:] if self.memory.conversations else []
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
                temperature=0.9
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Store conversation
            self.memory.add_conversation(user_input, ai_response)
            
            logger.info(f"🤖 NEXUS: '{ai_response}'")
            return ai_response
            
        except Exception as e:
            logger.error(f"❌ OpenAI error: {e}")
            return "Oh no! My brain just glitched for a second! Can you try that again? 😅"


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
    """Main NEXUS system"""
    
    def __init__(self):
        self.memory = NEXUSMemory()
        self.personality = NEXUSPersonality(self.memory)
        self.voice = NEXUSVoice()
        logger.info("🌟 NEXUS fully initialized and ready!")


# --- INITIALIZE NEXUS ---
nexus = NEXUS()
active_connections = {}

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
        "message": "ready",
        "nexus_ready": True
    }))
    
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    msg_type = data.get('type')
                    
                    if msg_type == 'text_input':
                        # Handle text input
                        user_input = data.get('message', '').strip()
                        
                        if user_input:
                            await ws.send_str(json.dumps({"type": "status", "message": "thinking"}))
                            
                            # Get response from NEXUS
                            response = await nexus.personality.generate_response(user_input)
                            
                            if response:
                                # Send text response
                                await ws.send_str(json.dumps({
                                    "type": "text_response",
                                    "message": response
                                }))
                                
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
                                # Return to ready state
                                await ws.send_str(json.dumps({
                                    "type": "status",
                                    "message": "ready"
                                }))
                    
                    elif msg_type == 'toggle_mode':
                        # Toggle wake word mode
                        nexus.personality.is_continuous_mode = not nexus.personality.is_continuous_mode
                        logger.info(f"🎯 Continuous mode: {'ON' if nexus.personality.is_continuous_mode else 'OFF'}")
                        await ws.send_str(json.dumps({
                            "type": "mode_toggled",
                            "continuous_mode": nexus.personality.is_continuous_mode
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
    
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NEXUS - Your AI Best Friend</title>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff; 
            display: flex; 
            flex-direction: column; 
            justify-content: center; 
            align-items: center; 
            height: 100vh; 
            margin: 0; 
            padding: 20px;
            box-sizing: border-box;
        }
        
        .nexus-container {
            text-align: center;
            z-index: 10;
            max-width: 600px;
            width: 100%;
            padding: 2rem;
            background: rgba(255,255,255,0.1);
            border-radius: 20px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        
        .nexus-logo {
            font-size: 4rem;
            font-weight: bold;
            margin-bottom: 1rem;
            text-shadow: 0 0 30px rgba(255,255,255,0.5);
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .status { 
            font-size: 1.5rem; 
            margin-bottom: 2rem;
            opacity: 0.9;
        }
        
        .chat-container {
            max-height: 300px;
            overflow-y: auto;
            background: rgba(0,0,0,0.2);
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            text-align: left;
        }
        
        .message {
            margin: 10px 0;
            padding: 10px 15px;
            border-radius: 15px;
            max-width: 80%;
        }
        
        .user-message {
            background: rgba(255,255,255,0.2);
            margin-left: auto;
            text-align: right;
        }
        
        .nexus-message {
            background: rgba(78,205,196,0.3);
            margin-right: auto;
        }
        
        .input-container {
            display: flex;
            gap: 10px;
            margin: 20px 0;
        }
        
        .text-input {
            flex: 1;
            padding: 15px;
            border: none;
            border-radius: 25px;
            background: rgba(255,255,255,0.2);
            color: white;
            font-size: 1rem;
            backdrop-filter: blur(10px);
        }
        
        .text-input::placeholder {
            color: rgba(255,255,255,0.7);
        }
        
        .send-btn {
            padding: 15px 25px;
            border: none;
            border-radius: 25px;
            background: #4ecdc4;
            color: #333;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .send-btn:hover {
            background: #45b7d1;
            transform: translateY(-2px);
        }
        
        .send-btn:disabled {
            background: #666;
            cursor: not-allowed;
            transform: none;
        }
        
        .mode-toggle {
            margin-top: 20px;
            padding: 10px 20px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 25px;
            background: rgba(255,255,255,0.1);
            color: white;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .mode-toggle.active {
            background: #4ecdc4;
            color: #333;
            border-color: #4ecdc4;
        }
        
        .memory-info {
            position: fixed;
            top: 20px;
            left: 20px;
            font-size: 0.9rem;
            opacity: 0.7;
            background: rgba(0,0,0,0.3);
            padding: 10px 15px;
            border-radius: 20px;
            backdrop-filter: blur(10px);
        }
        
        @media (max-width: 768px) {
            .nexus-logo { font-size: 2.5rem; }
            .nexus-container { padding: 1rem; margin: 10px; }
            .input-container { flex-direction: column; }
        }
    </style>
</head>
<body>
    <div class="memory-info" id="memoryInfo">
        🧠 Memories: 0 | 💬 Conversations: 0
    </div>
    
    <div class="nexus-container">
        <div class="nexus-logo">NEXUS</div>
        <div class="status" id="statusText">Your AI Best Friend is Ready!</div>
        
        <div class="chat-container" id="chatContainer">
            <div class="nexus-message">
                Hi! I'm NEXUS, your AI best friend! 💕<br>
                Say "Hey Nexus" to activate me, or just type below to chat!
            </div>
        </div>
        
        <div class="input-container">
            <input type="text" id="textInput" class="text-input" placeholder="Type your message here..." maxlength="500">
            <button id="sendBtn" class="send-btn">Send</button>
        </div>
        
        <button id="modeToggle" class="mode-toggle">
            Wake Word Mode: ON
        </button>
    </div>
    
    <script>
        let ws, reconnectAttempts = 0;
        let continuousMode = false;
        let audioQueue = [], isPlayingFromQueue = false;

        const statusText = document.getElementById('statusText');
        const chatContainer = document.getElementById('chatContainer');
        const textInput = document.getElementById('textInput');
        const sendBtn = document.getElementById('sendBtn');
        const modeToggle = document.getElementById('modeToggle');
        const memoryInfo = document.getElementById('memoryInfo');

        function addMessage(content, isUser = false) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user-message' : 'nexus-message'}`;
            messageDiv.textContent = content;
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function updateStatus(text) {
            statusText.textContent = text;
        }

        async function playAudioFromQueue() {
            if (isPlayingFromQueue || audioQueue.length === 0) return;
            isPlayingFromQueue = true;
            
            const data = audioQueue.shift();
            try {
                const audioData = atob(data);
                const buffer = new Uint8Array(audioData.length);
                for (let i = 0; i < audioData.length; i++) {
                    buffer[i] = audioData.charCodeAt(i);
                }
                
                const blob = new Blob([buffer], { type: 'audio/mpeg' });
                const url = URL.createObjectURL(blob);
                const audio = new Audio(url);
                
                audio.onended = () => {
                    URL.revokeObjectURL(url);
                    isPlayingFromQueue = false;
                    setTimeout(playAudioFromQueue, 100);
                };
                
                audio.onerror = () => {
                    URL.revokeObjectURL(url);
                    isPlayingFromQueue = false;
                    setTimeout(playAudioFromQueue, 100);
                };
                
                await audio.play();
                
            } catch (e) {
                console.error("Audio playback error:", e);
                isPlayingFromQueue = false;
                setTimeout(playAudioFromQueue, 100);
            }
        }

        function sendMessage() {
            const message = textInput.value.trim();
            if (!message || !ws || ws.readyState !== WebSocket.OPEN) return;
            
            addMessage(message, true);
            textInput.value = '';
            sendBtn.disabled = true;
            
            ws.send(JSON.stringify({
                type: 'text_input',
                message: message
            }));
        }

        function toggleMode() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'toggle_mode' }));
            }
        }

        // Event listeners
        sendBtn.addEventListener('click', sendMessage);
        modeToggle.addEventListener('click', toggleMode);
        
        textInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            console.log("🌐 Connecting to NEXUS:", wsUrl);
            ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                console.log("✅ Connected to NEXUS");
                reconnectAttempts = 0;
                updateStatus('Connected! Ready to chat! 💕');
                sendBtn.disabled = false;
            };
            
            ws.onmessage = (event) => {
                try {
                    const msg = JSON.parse(event.data);
                    console.log("📨 Received:", msg.type);
                    
                    if (msg.type === 'text_response') {
                        addMessage(msg.message);
                        sendBtn.disabled = false;
                    } else if (msg.type === 'audio_chunk') {
                        audioQueue.push(msg.data);
                        if (!isPlayingFromQueue) {
                            playAudioFromQueue();
                        }
                    } else if (msg.type === 'audio_stop') {
                        const checkDone = setInterval(() => {
                            if (audioQueue.length === 0 && !isPlayingFromQueue) {
                                clearInterval(checkDone);
                                updateStatus('Ready to chat! What\'s on your mind?');
                            }
                        }, 200);
                    } else if (msg.type === 'status') {
                        const statusMap = {
                            'ready': 'Ready to chat! 💕',
                            'thinking': 'NEXUS is thinking... 🤔',
                            'speaking': 'NEXUS is talking... 🗣️'
                        };
                        updateStatus(statusMap[msg.message] || msg.message);
                        
                        if (msg.message === 'ready') {
                            sendBtn.disabled = false;
                        }
                    } else if (msg.type === 'mode_toggled') {
                        continuousMode = msg.continuous_mode;
                        modeToggle.textContent = `Wake Word Mode: ${continuousMode ? 'OFF' : 'ON'}`;
                        modeToggle.classList.toggle('active', !continuousMode);
                    }
                } catch (e) {
                    console.error("❌ Message parsing error:", e);
                }
            };
            
            ws.onclose = () => {
                console.log("🔌 NEXUS disconnected");
                updateStatus('Connection lost... Reconnecting...');
                sendBtn.disabled = true;
                
                if (reconnectAttempts < 5) {
                    const delay = Math.pow(2, reconnectAttempts) * 1000;
                    setTimeout(() => {
                        reconnectAttempts++;
                        connect();
                    }, delay);
                } else {
                    updateStatus('Connection failed. Please refresh the page.');
                }
            };
            
            ws.onerror = (error) => {
                console.error("❌ WebSocket error:", error);
            };
        }

        // Initialize
        window.addEventListener('load', () => {
            console.log("🚀 Starting NEXUS...");
            connect();
        });

        // Update memory info periodically
        let memoryCounter = 0;
        setInterval(() => {
            memoryCounter++;
            memoryInfo.textContent = `🧠 Memories: ${memoryCounter} | 💬 Active: ${ws && ws.readyState === WebSocket.OPEN ? 'Yes' : 'No'}`;
        }, 10000);
    </script>
</body>
</html>
"""
    return web.Response(text=html_content, content_type='text/html')


# --- APPLICATION SETUP ---
def create_app():
    """Create the NEXUS web application"""
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
    app.router.add_get('/', serve_interface)
    app.router.add_get('/ws', websocket_handler)
    
    # Add CORS to all routes
    for route in list(app.router.routes()):
        cors.add(route)
    
    return app


# --- MAIN APPLICATION ---
if __name__ == "__main__":
    # Validate environment
    if not os.getenv('OPENAI_API_KEY'):
        logger.error("❌ OPENAI_API_KEY environment variable required!")
        exit(1)
    
    # Create app
    app = create_app()
    
    logger.info("🌟 NEXUS - Your AI Best Friend (Ultra Minimal)")
    logger.info(f"🌐 Starting server on http://{HOST}:{PORT}")
    logger.info("💕 NEXUS is ready to be your companion 24/7!")
    
    # Run the app
    web.run_app(app, host=HOST, port=PORT)
