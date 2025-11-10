#app.py

import os
import json
import asyncio
from fastapi import FastAPI, WebSocket, UploadFile, File, Form, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn
import pyttsx3
import tempfile
import subprocess
import base64

from utils import ensure_session, save_temp_file, cleanup_file, session_memories

import whisper
from langchain_community.llms import Ollama 

# Configure via env
#TGI_URL = os.environ.get('TGI_URL')
#TGI_API_KEY = os.environ.get('TGI_API_KEY')
WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "medium")
#TGI_URL = os.environ.get("TGI_URL")
VOICE_DEFAULT = os.environ.get("VOICE_DEFAULT", "default")

print("Loading Whisper model:", WHISPER_MODEL_NAME)
whisper_model = whisper.load_model(WHISPER_MODEL_NAME)

# instantiate the LangChain wrapper for Ollama
llm = Ollama(model="llama3")

app = FastAPI()

async def transcribe_file(path: str) -> str:
    loop = asyncio.get_running_loop()
    def _transcribe():
        # whisper returns dict with 'text'
        r = whisper_model.transcribe(path)
        return r.get('text', '')
    return await loop.run_in_executor(None, _transcribe)

# Simple upload endpoint for file-based flow (non-streaming)
@app.post('/upload_audio')
async def upload_audio(file: UploadFile = File(...), session_id: str = Form(None), speaker_id: str = Form('default')):
    data = await file.read()
    # Save temp file
    path = save_temp_file(data, suffix='.webm')
    try:
        transcription = await transcribe_file(path)
    finally:
        cleanup_file(path)

    session_id = ensure_session(session_id)
    session_memories[session_id].append({'role':'User','text':transcription})

    # Get LLM non-streamed result (for simple clients)
    # Here we just collect tokens from stream and concatenate
    result_parts = []
    async for tok in stream_llm_tokens(transcription, max_tokens=256):
        result_parts.append(tok)
    llm_response = ''.join(result_parts)
    session_memories[session_id].append({'role':'Assistant','text':llm_response})
    return JSONResponse({'session_id':session_id,'transcription':transcription,'response':llm_response})

async def synthesize_tts_and_stream(websocket, text):
    """Generate valid WAV with pyttsx3 and send it as one binary message."""
    tts_engine = pyttsx3.init()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    # Generate WAV file
    tts_engine.save_to_file(text, wav_path)
    tts_engine.runAndWait()
    tts_engine.stop()

    # Convert WAV to browser-friendly PCM WAV using ffmpeg
    converted_path = wav_path + "_conv.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", wav_path,
        "-ar", "22050", "-ac", "1", "-f", "wav", converted_path
    ], check=True)

    # Send the converted file as binary
    with open(converted_path, "rb") as f:
        await websocket.send_bytes(f.read())

    # Clean up
    os.remove(wav_path)
    os.remove(converted_path)

@app.websocket('/ws/{session_id}')
async def ws_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    try:
        # Expect initial JSON start message
        msg = await websocket.receive_text()
        msg_json = json.loads(msg)
        speaker_id = msg_json.get('speaker_id','default')

        # The client can choose to send either `prompt` directly or upload audio first
        # (we'll allow prompt)
        prompt = msg_json.get('prompt')
        audio_blob_b64 = msg_json.get('audio_b64') # optional base64 audio field

        if audio_blob_b64 and not prompt:
            # decode/save/transcribe 
            data = base64.b64decode(audio_blob_b64)
            path = save_temp_file(data, suffix='.webm')
            try:
                prompt = await transcribe_file(path)
            finally:
                cleanup_file(path)

        if not prompt:
            await websocket.send_json({'type':'error','message':'no prompt provided'})
            #return
        
        session_memories[session_id].append({'role':'User','text':prompt})
        await websocket.send_json({'type':'prompt','message':prompt})

        # produce a response using LLM (Llama3) based on prompt
        if prompt:
            try:
                # for LangChain 1.x, .invoke() expects a string
                response = str(llm.invoke(prompt))  
                await websocket.send_json({'type':'response','message':response})
            except Exception as e:
                await websocket.send_json({'type':'error','message':str(e)})
                return f"⚠️ Calling Llama3 error: {e}"

        # Start TTS: notify client, then send binary chunks
        await websocket.send_json({'type':'tts_start'})
        await synthesize_tts_and_stream(websocket, response)
        await websocket.send_json({'type':'tts_end'})

    except WebSocketDisconnect:
        print('client disconnected')
    except Exception as e:
        await websocket.send_json({'type':'error','message':str(e)})
        raise
    
if __name__ == '__main__':
    uvicorn.run('backend.app:app', host='0.0.0.0', port=8000, reload=True)