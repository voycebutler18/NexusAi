import os
from dotenv import load_dotenv
import openai
import asyncio
import websockets
import time
import base64
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import io
from aiohttp import web, WSMsgType, web_ws
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
    logger.error("📝 Please set the OPENAI_API_KEY environment variable")
    raise ValueError("OPENAI_API_KEY environment variable is required")

logger.info("✅ OpenAI API key loaded securely")

# Get configuration from environment
PORT = int(os.getenv('PORT', 8080))
HOST = os.getenv('HOST', '0.0.0.0')

# --- GLOBAL STATE ---
active_connections = set()
conversation_context = []
current_visual_context = ""
executor = ThreadPoolExecutor(max_workers=4)

# --- WEBSOCKET HANDLERS ---
async def broadcast_to_all(message):
    """Broadcast message to all active connections."""
    if not active_connections:
        return
    
    disconnected = set()
    for ws in active_connections.copy():
        try:
            await ws.send_str(message)
        except Exception as e:
            logger.warning(f"Failed to send to connection: {e}")
            disconnected.add(ws)
    
    # Remove disconnected connections
    active_connections.difference_update(disconnected)

async def broadcast_json(data):
    """Broadcast JSON data to all connections."""
    message = json.dumps(data)
    await broadcast_to_all(message)

# --- AI FUNCTIONS ---
async def get_text_response(user_input):
    """Get response from OpenAI."""
    global conversation_context, current_visual_context
    
    try:
        logger.info(f"Processing: {user_input[:50]}...")
        
        system_prompt = f"""You are NEXUS 3000, an advanced AI assistant.

Visual context: {current_visual_context if current_visual_context else "None"}

Respond naturally and helpfully. Be engaging and conversational."""

        conversation_context.append({"role": "user", "content": user_input})
        if len(conversation_context) > 12:
            conversation_context = conversation_context[-12:]
        
        messages = [{"role": "system", "content": system_prompt}] + conversation_context
        
        response = await asyncio.get_event_loop().run_in_executor(
            executor, lambda: openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=200,
                temperature=0.7
            )
        )
        
        ai_response = response.choices[0].message.content.strip()
        conversation_context.append({"role": "assistant", "content": ai_response})
        
        return ai_response
        
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        return "I'm experiencing some technical difficulties right now. Please try again."

async def analyze_image(image_data):
    """Analyze uploaded image."""
    global current_visual_context
    
    try:
        logger.info("Analyzing image...")
        
        response = await asyncio.get_event_loop().run_in_executor(
            executor, lambda: openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are NEXUS 3000. Analyze images and describe what you see clearly and helpfully."
                    },
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": "What do you see in this image?"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                        ]
                    }
                ],
                max_tokens=150,
                temperature=0.7
            )
        )
        
        analysis = response.choices[0].message.content.strip()
        current_visual_context = analysis
        
        logger.info(f"Image analysis complete: {analysis[:50]}...")
        return analysis
        
    except Exception as e:
        logger.error(f"Vision analysis error: {e}")
        return "I'm having trouble analyzing this image right now."

async def transcribe_audio(audio_data):
    """Transcribe audio using OpenAI Whisper."""
    try:
        logger.info("Transcribing audio...")
        
        # Convert base64 audio to bytes
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
        logger.info(f"Transcribed: {text}")
        return text
        
    except Exception as e:
        logger.error(f"Audio transcription error: {e}")
        return ""

# --- HTTP HANDLERS ---
async def websocket_handler(request):
    """Handle WebSocket connections."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    active_connections.add(ws)
    client_addr = request.remote
    logger.info(f"🌐 New connection from {client_addr}")
    
    # Send welcome message
    welcome_msg = {
        "type": "status",
        "message": "Connected to NEXUS 3000",
        "timestamp": time.time()
    }
    await ws.send_str(json.dumps(welcome_msg))
    
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    message_type = data.get('type')
                    
                    if message_type == 'text':
                        user_text = data.get('message', '').strip()
                        if user_text:
                            # Broadcast user message
                            await broadcast_json({
                                "type": "user_message",
                                "message": user_text,
                                "timestamp": time.time()
                            })
                            
                            # Get AI response
                            ai_response = await get_text_response(user_text)
                            await broadcast_json({
                                "type": "ai_response",
                                "message": ai_response,
                                "timestamp": time.time()
                            })
                    
                    elif message_type == 'image':
                        image_data = data.get('data', '')
                        if image_data:
                            analysis = await analyze_image(image_data)
                            await broadcast_json({
                                "type": "image_analysis",
                                "message": analysis,
                                "timestamp": time.time()
                            })
                    
                    elif message_type == 'audio':
                        audio_data = data.get('data', '')
                        if audio_data:
                            transcription = await transcribe_audio(audio_data)
                            if transcription:
                                await broadcast_json({
                                    "type": "audio_transcription",
                                    "message": transcription,
                                    "timestamp": time.time()
                                })
                                
                                # Get AI response to transcription
                                ai_response = await get_text_response(transcription)
                                await broadcast_json({
                                    "type": "ai_response",
                                    "message": ai_response,
                                    "timestamp": time.time()
                                })
                    
                    elif message_type == 'ping':
                        await ws.send_str(json.dumps({"type": "pong"}))
                        
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from {client_addr}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    
            elif msg.type == WSMsgType.ERROR:
                logger.error(f'WebSocket error: {ws.exception()}')
                break
                
    except Exception as e:
        logger.error(f"WebSocket error with {client_addr}: {e}")
    finally:
        active_connections.discard(ws)
        logger.info(f"🔌 Client {client_addr} disconnected")
    
    return ws

async def health_check(request):
    """Health check endpoint."""
    return web.Response(text="NEXUS 3000 is online!", status=200)

async def index_handler(request):
    """Serve the main interface."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NEXUS 3000 • Web Interface</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #16213e 100%);
            color: white;
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .title {
            text-align: center;
            font-size: 3em;
            margin-bottom: 30px;
            background: linear-gradient(45deg, #00d4ff, #ff006e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .chat-container {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            height: 400px;
            overflow-y: auto;
            border: 2px solid #00d4ff;
        }
        .message {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 10px;
        }
        .user-message {
            background: rgba(0, 212, 255, 0.2);
            text-align: right;
        }
        .ai-message {
            background: rgba(255, 0, 110, 0.2);
        }
        .input-area {
            display: flex;
            gap: 10px;
        }
        .text-input {
            flex: 1;
            padding: 15px;
            border: 2px solid #00d4ff;
            border-radius: 10px;
            background: rgba(0, 0, 0, 0.5);
            color: white;
            font-size: 16px;
        }
        .send-btn {
            padding: 15px 30px;
            background: linear-gradient(45deg, #00d4ff, #ff006e);
            border: none;
            border-radius: 10px;
            color: white;
            font-weight: bold;
            cursor: pointer;
        }
        .status {
            text-align: center;
            margin: 10px 0;
            font-style: italic;
            color: #00d4ff;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="title">NEXUS 3000</h1>
        <div class="status" id="status">Connecting...</div>
        <div class="chat-container" id="chatContainer"></div>
        <div class="input-area">
            <input type="text" class="text-input" id="textInput" placeholder="Type your message..." onkeypress="handleKeyPress(event)">
            <button class="send-btn" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        let ws;
        const chatContainer = document.getElementById('chatContainer');
        const textInput = document.getElementById('textInput');
        const status = document.getElementById('status');

        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function() {
                status.textContent = 'Connected to NEXUS 3000';
                status.style.color = '#00ff00';
            };
            
            ws.onmessage = function(event) {
                try {
                    const data = JSON.parse(event.data);
                    handleMessage(data);
                } catch (e) {
                    console.error('Error parsing message:', e);
                }
            };
            
            ws.onclose = function() {
                status.textContent = 'Disconnected - Reconnecting...';
                status.style.color = '#ff0000';
                setTimeout(connectWebSocket, 3000);
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
        }

        function handleMessage(data) {
            if (data.type === 'user_message') {
                addMessage(data.message, 'user');
            } else if (data.type === 'ai_response') {
                addMessage(data.message, 'ai');
            } else if (data.type === 'status') {
                status.textContent = data.message;
            }
        }

        function addMessage(message, type) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}-message`;
            messageDiv.textContent = message;
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function sendMessage() {
            const message = textInput.value.trim();
            if (message && ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'text',
                    message: message
                }));
                textInput.value = '';
            }
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }

        // Connect when page loads
        window.addEventListener('load', connectWebSocket);
    </script>
</body>
</html>"""
    return web.Response(text=html_content, content_type='text/html')

# --- MAIN APPLICATION ---
def create_app():
    """Create the web application."""
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
    
    # Add routes
    app.router.add_get('/', index_handler)
    app.router.add_get('/health', health_check)
    app.router.add_get('/ws', websocket_handler)
    
    # Add CORS to all routes
    for route in list(app.router.routes()):
        cors.add(route)
    
    return app

async def main():
    """Main application entry point."""
    logger.info("🚀 Starting NEXUS 3000...")
    logger.info(f"🌐 Host: {HOST}:{PORT}")
    
    try:
        app = create_app()
        
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, HOST, PORT)
        await site.start()
        
        logger.info("✅ NEXUS 3000 is online!")
        logger.info(f"🌐 Access at: http://{HOST}:{PORT}")
        
        # Keep the server running
        while True:
            await asyncio.sleep(3600)
            
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 NEXUS 3000 shutting down...")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        raise
