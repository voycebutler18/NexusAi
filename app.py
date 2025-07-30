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
import aiohttp

# Load environment variables from .env file
load_dotenv()

# --- SECURE CONFIGURATION ---
openai.api_key = os.getenv('OPENAI_API_KEY')

# Validate API key is loaded
if not openai.api_key:
    raise ValueError("❌ OpenAI API key not found! Please set OPENAI_API_KEY in your .env file.")

print("✅ OpenAI API key loaded securely from environment")

# Get port from environment (Render sets this automatically)
PORT = int(os.getenv('PORT', 8765))
HOST = os.getenv('HOST', '0.0.0.0')

# --- GLOBAL STATE ---
clients = set()
conversation_context = []
current_visual_context = ""
last_vision_analysis = ""
vision_analysis_active = False

executor = ThreadPoolExecutor(max_workers=4)

# --- WEBSOCKET BROADCAST ---
async def broadcast(message):
    """Send message to all WebSocket clients."""
    if clients:
        disconnected = set()
        for client in clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                print(f"❌ Broadcast error: {e}")
                disconnected.add(client)
        
        # Remove disconnected clients
        clients.difference_update(disconnected)

async def broadcast_json(data):
    """Send JSON data to all WebSocket clients."""
    message = json.dumps(data)
    await broadcast(message)

# --- AI RESPONSE FUNCTIONS ---
async def get_text_response(user_input):
    """Get response from OpenAI for text input."""
    global conversation_context, current_visual_context
    
    try:
        print("🧠 Getting AI response...")
        
        system_prompt = f"""You are NEXUS, an AI assistant. 

Visual context: {current_visual_context if current_visual_context else "None"}

Respond naturally and conversationally. Be helpful and engaging."""

        conversation_context.append({"role": "user", "content": user_input})
        if len(conversation_context) > 10:
            conversation_context = conversation_context[-10:]
        
        messages = [
            {"role": "system", "content": system_prompt}
        ] + conversation_context
        
        response = await asyncio.get_event_loop().run_in_executor(
            executor, lambda: openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=150,
                temperature=0.7
            )
        )
        
        ai_response = response.choices[0].message.content.strip()
        conversation_context.append({"role": "assistant", "content": ai_response})
        
        return ai_response
        
    except Exception as e:
        print(f"❌ OpenAI error: {e}")
        return "I apologize, but I'm experiencing some technical difficulties right now."

async def analyze_image(image_data):
    """Analyze uploaded image."""
    global current_visual_context
    
    try:
        print("👁️ Analyzing image...")
        
        response = await asyncio.get_event_loop().run_in_executor(
            executor, lambda: openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are NEXUS. Analyze this image and describe what you see. Be detailed but concise."
                    },
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": "What do you see in this image?"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                        ]
                    }
                ],
                max_tokens=100,
                temperature=0.7
            )
        )
        
        analysis = response.choices[0].message.content.strip()
        current_visual_context = analysis
        
        print(f"👁️ NEXUS sees: {analysis}")
        return analysis
        
    except Exception as e:
        print(f"❌ Vision analysis error: {e}")
        return "I'm having trouble analyzing this image right now."

async def transcribe_audio(audio_data):
    """Transcribe audio using OpenAI Whisper."""
    try:
        print("🎤 Transcribing audio...")
        
        # Convert base64 audio to file-like object
        audio_bytes = base64.b64decode(audio_data.split(',')[1])
        
        # Create a temporary file-like object
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.webm"  # Whisper needs a filename
        
        response = await asyncio.get_event_loop().run_in_executor(
            executor, lambda: openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        )
        
        text = response.text.strip()
        print(f"🗣️ Transcribed: '{text}'")
        return text
        
    except Exception as e:
        print(f"❌ Audio transcription error: {e}")
        return ""

# --- WEBSOCKET HANDLER ---
async def handle_websocket(websocket, path):
    """Handle WebSocket connections and messages."""
    clients.add(websocket)
    client_address = websocket.remote_address
    print(f"🌐 Client connected: {client_address}")
    
    try:
        # Send welcome message
        welcome_msg = {
            "type": "status",
            "message": "Connected to NEXUS",
            "timestamp": time.time()
        }
        await websocket.send(json.dumps(welcome_msg))
        
        async for message in websocket:
            try:
                data = json.loads(message)
                message_type = data.get('type')
                
                if message_type == 'text':
                    # Handle text message
                    user_text = data.get('message', '')
                    if user_text:
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
                    # Handle image upload
                    image_data = data.get('data', '')
                    if image_data:
                        analysis = await analyze_image(image_data)
                        await broadcast_json({
                            "type": "image_analysis",
                            "message": analysis,
                            "timestamp": time.time()
                        })
                
                elif message_type == 'audio':
                    # Handle audio upload
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
                    # Handle ping/keepalive
                    await websocket.send(json.dumps({"type": "pong"}))
                    
            except json.JSONDecodeError:
                print(f"❌ Invalid JSON from {client_address}")
            except Exception as e:
                print(f"❌ Error handling message from {client_address}: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        print(f"🔌 Client disconnected: {client_address}")
    except Exception as e:
        print(f"❌ WebSocket error with {client_address}: {e}")
    finally:
        clients.discard(websocket)

# --- HTTP SERVER FOR HEALTH CHECKS ---
async def health_check(request):
    """Health check endpoint for Render."""
    return web.Response(text="NEXUS is running!", status=200)

async def create_http_server():
    """Create HTTP server for health checks."""
    from aiohttp import web
    
    app = web.Application()
    app.router.add_get('/', health_check)
    app.router.add_get('/health', health_check)
    
    # Serve static files if they exist
    try:
        app.router.add_static('/', path='./static', name='static')
    except:
        pass
    
    return app

# --- MAIN PROGRAM ---
async def main():
    """Main program entry point."""
    print("🚀 Starting NEXUS Web Server...")
    print(f"🌐 Host: {HOST}:{PORT}")
    
    try:
        # Start WebSocket server
        print(f"🔌 Starting WebSocket server on {HOST}:{PORT}")
        
        # Start the WebSocket server
        start_server = websockets.serve(
            handle_websocket, 
            HOST, 
            PORT,
            ping_interval=20,
            ping_timeout=10
        )
        
        print("✅ NEXUS Web Server is running!")
        print(f"🌐 WebSocket endpoint: ws://{HOST}:{PORT}")
        
        # Keep the server running
        await start_server
        
        # Run forever
        while True:
            await asyncio.sleep(3600)  # Sleep for 1 hour, then check again
            
    except Exception as e:
        print(f"❌ Server error: {e}")
        raise

if __name__ == "__main__":
    print("🔧 Initializing NEXUS Web Server...")
    print("✅ All systems ready")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 NEXUS shutting down...")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
    finally:
        print("✅ NEXUS shutdown complete.")
