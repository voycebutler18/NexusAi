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

# --- HTTP HANDLERS ---
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
    <title>NEXUS 3000 • AI Assistant</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #16213e 100%);
            color: white;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            width: 100%;
            max-width: 900px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 20px;
            padding: 30px;
            backdrop-filter: blur(10px);
            border: 2px solid rgba(0, 212, 255, 0.3);
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
        }
        
        .title {
            text-align: center;
            font-size: 3.5em;
            font-weight: 900;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #00d4ff, #ff006e, #8338ec);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
        }
        
        .subtitle {
            text-align: center;
            color: #00d4ff;
            margin-bottom: 30px;
            font-size: 1.2em;
            letter-spacing: 0.2em;
        }
        
        .status {
            text-align: center;
            margin-bottom: 20px;
            padding: 10px;
            border-radius: 10px;
            background: rgba(0, 212, 255, 0.1);
            border: 1px solid #00d4ff;
            font-weight: 500;
        }
        
        .chat-container {
            background: rgba(0, 0, 0, 0.4);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            height: 400px;
            overflow-y: auto;
            border: 2px solid rgba(0, 212, 255, 0.2);
        }
        
        .message {
            margin-bottom: 15px;
            padding: 15px;
            border-radius: 15px;
            line-height: 1.6;
        }
        
        .user-message {
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.2), rgba(0, 245, 255, 0.1));
            margin-left: 50px;
            border-left: 4px solid #00d4ff;
        }
        
        .ai-message {
            background: linear-gradient(135deg, rgba(255, 0, 110, 0.2), rgba(131, 56, 236, 0.1));
            margin-right: 50px;
            border-left: 4px solid #ff006e;
        }
        
        .input-container {
            display: flex;
            gap: 15px;
            align-items: center;
        }
        
        .text-input {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid rgba(0, 212, 255, 0.3);
            border-radius: 15px;
            background: rgba(0, 0, 0, 0.5);
            color: white;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        
        .text-input:focus {
            outline: none;
            border-color: #00d4ff;
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
        }
        
        .send-btn {
            padding: 15px 30px;
            background: linear-gradient(45deg, #00d4ff, #ff006e);
            border: none;
            border-radius: 15px;
            color: white;
            font-weight: bold;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .send-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(0, 212, 255, 0.4);
        }
        
        .send-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        /* Scrollbar styling */
        .chat-container::-webkit-scrollbar {
            width: 8px;
        }
        
        .chat-container::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
        }
        
        .chat-container::-webkit-scrollbar-thumb {
            background: linear-gradient(45deg, #00d4ff, #ff006e);
            border-radius: 10px;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .container {
                padding: 20px;
                margin: 10px;
            }
            
            .title {
                font-size: 2.5em;
            }
            
            .user-message {
                margin-left: 20px;
            }
            
            .ai-message {
                margin-right: 20px;
            }
            
            .input-container {
                flex-direction: column;
                gap: 10px;
            }
            
            .text-input, .send-btn {
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="title">NEXUS 3000</h1>
        <p class="subtitle">Advanced AI Assistant</p>
        
        <div class="status" id="status">Connecting to NEXUS 3000...</div>
        
        <div class="chat-container" id="chatContainer">
            <div class="message ai-message">
                Welcome to NEXUS 3000! I'm your advanced AI assistant. How can I help you today?
            </div>
        </div>
        
        <div class="input-container">
            <input 
                type="text" 
                class="text-input" 
                id="textInput" 
                placeholder="Type your message here..." 
                onkeypress="handleKeyPress(event)"
            >
            <button class="send-btn" id="sendBtn" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        let ws;
        let isConnected = false;
        
        const chatContainer = document.getElementById('chatContainer');
        const textInput = document.getElementById('textInput');
        const sendBtn = document.getElementById('sendBtn');
        const status = document.getElementById('status');

        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            status.textContent = 'Connecting to NEXUS 3000...';
            status.style.background = 'rgba(255, 165, 0, 0.1)';
            status.style.borderColor = '#ffa500';
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function() {
                isConnected = true;
                status.textContent = '🌐 Connected to NEXUS 3000 - Ready for interaction';
                status.style.background = 'rgba(0, 255, 0, 0.1)';
                status.style.borderColor = '#00ff00';
                sendBtn.disabled = false;
                textInput.disabled = false;
                textInput.focus();
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
                isConnected = false;
                status.textContent = '🔴 Connection lost - Reconnecting...';
                status.style.background = 'rgba(255, 0, 0, 0.1)';
                status.style.borderColor = '#ff0000';
                sendBtn.disabled = true;
                textInput.disabled = true;
                setTimeout(connectWebSocket, 3000);
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
                status.textContent = '❌ Connection error - Retrying...';
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
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }

        // Initialize connection when page loads
        window.addEventListener('load', function() {
            connectWebSocket();
        });

        // Handle page visibility for reconnection
        document.addEventListener('visibilitychange', function() {
            if (document.visibilityState === 'visible' && !isConnected) {
                connectWebSocket();
            }
        });
    </script>
</body>
</html>"""
    return web.Response(text=html_content, content_type='text/html')

# --- APPLICATION SETUP ---
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
    logger.info(f"🌐 Server will run on {HOST}:{PORT}")
    
    try:
        app = create_app()
        
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, HOST, PORT)
        await site.start()
        
        logger.info("✅ NEXUS 3000 is online!")
        logger.info(f"🌐 Access your app at: http://{HOST}:{PORT}")
        
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
