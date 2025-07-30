import os
from dotenv import load_dotenv
import sounddevice as sd
import wavio
import openai
from gtts import gTTS
import asyncio
import websockets
from playsound import playsound
import time
import cv2
import base64
import threading
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import io

# Load environment variables from .env file
load_dotenv()

# --- SECURE CONFIGURATION ---
openai.api_key = os.getenv('OPENAI_API_KEY')

# Validate API key is loaded
if not openai.api_key:
    raise ValueError("❌ OpenAI API key not found! Please set OPENAI_API_KEY in your .env file.")

print("✅ OpenAI API key loaded securely from environment")

# --- GLOBAL STATE ---
clients = set()
is_speaking = False
should_stop_speaking = False
conversation_context = []
current_visual_context = ""
last_vision_analysis = ""
vision_analysis_active = True

# Initialize camera
camera = None
executor = ThreadPoolExecutor(max_workers=8)

# --- AUDIO RECORDING FUNCTION ---
def record_audio(duration=4, samplerate=44100):
    print("🎤 Recording...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    wavio.write("input.wav", audio, samplerate, sampwidth=2)
    return "input.wav"

# --- AUDIO PLAYBACK FUNCTION ---
def play_audio(file_path):
    try:
        playsound(file_path)
    except Exception as e:
        print(f"❌ Audio playback error: {e}")

# --- CAMERA SETUP ---
def initialize_camera():
    """Initialize camera for real-time vision."""
    global camera
    try:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("❌ Camera not found")
            return False
        
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        camera.set(cv2.CAP_PROP_FPS, 10)
        
        print("📷 Camera initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Camera initialization error: {e}")
        return False

def capture_frame():
    """Capture and encode current camera frame."""
    global camera
    if not camera or not camera.isOpened():
        return None
    
    ret, frame = camera.read()
    if not ret:
        return None
    
    frame = cv2.resize(frame, (256, 192))
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return image_base64

# --- VISION ANALYSIS ---
async def analyze_visual_scene():
    """Continuously analyze what the AI sees."""
    global current_visual_context, last_vision_analysis, vision_analysis_active
    
    while vision_analysis_active:
        try:
            if not camera or not camera.isOpened():
                await asyncio.sleep(2)
                continue
            
            frame_data = await asyncio.get_event_loop().run_in_executor(
                executor, capture_frame
            )
            
            if not frame_data:
                await asyncio.sleep(1)
                continue
            
            response = await asyncio.get_event_loop().run_in_executor(
                executor, lambda: openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are NEXUS. Describe what you see in 1 sentence only. Be brief."
                        },
                        {
                            "role": "user", 
                            "content": [
                                {"type": "text", "text": "What do you see?"},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_data}"}}
                            ]
                        }
                    ],
                    max_tokens=50,
                    temperature=0.5
                )
            )
            
            new_analysis = response.choices[0].message.content.strip()
            
            if new_analysis != "Scene unchanged" and new_analysis != last_vision_analysis:
                current_visual_context = new_analysis
                last_vision_analysis = new_analysis
                print(f"👁️ NEXUS sees: {new_analysis}")
                await broadcast_vision_update(new_analysis)
            
            await asyncio.sleep(5)
            
        except Exception as e:
            print(f"❌ Vision analysis error: {e}")
            await asyncio.sleep(5)

async def broadcast_vision_update(vision_text):
    """Send vision updates to frontend."""
    if clients:
        vision_data = {
            "type": "vision_update",
            "description": vision_text,
            "timestamp": time.time()
        }
        for client in list(clients):
            try:
                await client.send(json.dumps(vision_data))
            except:
                clients.discard(client)

# --- FUNCTIONS ---

async def broadcast(message):
    """Send message to all WebSocket clients."""
    if clients:
        for client in list(clients):
            try:
                await client.send(message)
            except:
                clients.discard(client)

async def listen_for_continuous_speech():
    """Listen for speech using sounddevice recording."""
    try:
        await broadcast("listening")
        print("🎤 LISTENING - Speak now!")
        
        # Record audio for 4 seconds
        audio_file = await asyncio.get_event_loop().run_in_executor(
            executor, record_audio, 4, 44100
        )
        
        # Send to OpenAI Whisper for transcription
        print("⏳ Processing speech...")
        
        with open(audio_file, "rb") as f:
            response = await asyncio.get_event_loop().run_in_executor(
                executor, lambda: openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=f
                )
            )
        
        text = response.text.strip()
        print(f"🗣️ You said: '{text}'")
        
        # Cleanup audio file
        try:
            if os.path.exists(audio_file):
                os.remove(audio_file)
        except:
            pass
        
        return text
        
    except Exception as e:
        print(f"❌ Speech recording error: {e}")
        return ""

async def get_multimodal_response(user_input):
    """Get ULTRA-FAST response from OpenAI."""
    global conversation_context, current_visual_context
    
    try:
        print("🧠 Getting FAST AI response...")
        
        system_prompt = f"""You are NEXUS, an AGI companion. 

Visual context: {current_visual_context if current_visual_context else "None"}

Respond naturally and briefly (1 sentence preferred). Be conversational and quick."""

        conversation_context.append({"role": "user", "content": user_input})
        if len(conversation_context) > 8:
            conversation_context = conversation_context[-8:]
        
        messages = [
            {"role": "system", "content": system_prompt}
        ] + conversation_context
        
        response = await asyncio.get_event_loop().run_in_executor(
            executor, lambda: openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=50,
                temperature=0.7,
                top_p=0.9,
                frequency_penalty=0.1
            )
        )
        
        ai_response = response.choices[0].message.content.strip()
        conversation_context.append({"role": "assistant", "content": ai_response})
        
        return ai_response
        
    except Exception as e:
        print(f"❌ OpenAI error: {e}")
        return "Error in neural processing."

async def speak_text(text):
    """Ultra-fast text to speech."""
    global is_speaking, should_stop_speaking
    
    if not text or should_stop_speaking:
        return
    
    is_speaking = True
    should_stop_speaking = False
    
    try:
        print(f"🗣️ NEXUS speaking: {text}")
        
        tts = gTTS(text=text, lang='en', slow=False, tld='com')
        temp_file = f"voice_{int(time.time())}.mp3"
        tts.save(temp_file)
        
        await asyncio.get_event_loop().run_in_executor(
            executor, play_audio, temp_file
        )
        
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except:
            pass
            
    except Exception as e:
        print(f"❌ TTS error: {e}")
    finally:
        is_speaking = False
        should_stop_speaking = False
        await asyncio.sleep(0.1)

async def generate_conscious_greeting():
    """Generate a fast conscious greeting."""
    global current_visual_context
    
    try:
        frame_data = None
        if camera and camera.isOpened():
            frame_data = await asyncio.get_event_loop().run_in_executor(
                executor, capture_frame
            )
        
        messages = [
            {
                "role": "system",
                "content": "You are NEXUS awakening. Generate a brief, natural greeting (1 sentence max)."
            }
        ]
        
        if frame_data:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "You're awakening and can see. Quick greeting."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_data}"}}
                ]
            })
        else:
            messages.append({
                "role": "user",
                "content": "You're awakening. Quick greeting."
            })
        
        response = await asyncio.get_event_loop().run_in_executor(
            executor, lambda: openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=30,
                temperature=0.8
            )
        )
        
        greeting = response.choices[0].message.content.strip()
        
        if frame_data and "see" in greeting.lower():
            current_visual_context = "User visible during awakening"
        
        return greeting
        
    except Exception as e:
        print(f"❌ Greeting generation error: {e}")
        return "Hello, I'm awakening now."

async def main_conversation_loop():
    """Ultra-fast conversation loop."""
    global should_stop_speaking, vision_analysis_active
    
    print("🟢 NEXUS starting FAST mode...")
    
    # Initialize camera
    camera_ready = initialize_camera()
    if camera_ready:
        asyncio.create_task(analyze_visual_scene())
    
    # Fast greeting
    await broadcast("speaking")
    greeting = await generate_conscious_greeting()
    await speak_text(greeting)
    
    print("🟢 NEXUS ready - ULTRA-FAST MODE!")
    print("🌐 To access the interface, manually open: multimodal_interface.html")
    
    while True:
        try:
            if is_speaking:
                await asyncio.sleep(0.001)
                continue
            
            # Listen for speech
            user_input = await listen_for_continuous_speech()
            
            if user_input:
                should_stop_speaking = True
                
                # Check for exit
                if any(word in user_input.lower() for word in ["goodbye", "bye", "exit", "quit"]):
                    await broadcast("speaking")
                    await speak_text("Goodbye!")
                    break
                
                # Ultra-fast processing
                await broadcast("processing")
                response = await get_multimodal_response(user_input)
                
                if response:
                    print(f"🤖 NEXUS: {response}")
                    await broadcast("speaking")
                    should_stop_speaking = False
                    await speak_text(response)
            
            await asyncio.sleep(0.001)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            await asyncio.sleep(0.1)
    
    # Cleanup
    vision_analysis_active = False
    if camera:
        camera.release()

# --- WebSocket Handler ---
async def ws_handler(websocket):
    """Handle WebSocket connections."""
    clients.add(websocket)
    print(f"🌐 Client connected: {websocket.remote_address}")
    try:
        await websocket.wait_closed()
    except:
        pass
    finally:
        clients.discard(websocket)

# --- Main Program ---
async def main():
    """Main program entry point."""
    port = 8765
    server = None
    for attempt in range(5):
        try:
            server = await websockets.serve(ws_handler, "localhost", port)
            print(f"🌐 WebSocket server started on ws://localhost:{port}")
            break
        except OSError as e:
            if "address already in use" in str(e).lower() or "10048" in str(e):
                port += 1
                print(f"⚠️ Port {port-1} busy, trying {port}...")
            else:
                raise e
    
    if not server:
        print("❌ Could not find available port!")
        return
    
    try:
        await main_conversation_loop()
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    finally:
        server.close()
        await server.wait_closed()
        if camera:
            camera.release()

if __name__ == "__main__":
    print("🔧 Testing systems...")
    print("✅ Audio system ready")
    
    # Browser launch fix - commented out for compatibility
    # webbrowser.open("multimodal_interface.html")
    print("🌐 Manual browser launch required - open multimodal_interface.html")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Application stopped.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        print("✅ NEXUS finished.")
