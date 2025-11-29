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

import math
import sympy as sp  
import requests
import arxiv

# Configure via env
WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "medium")
VOICE_DEFAULT = os.environ.get("VOICE_DEFAULT", "default")
ARXIV_MAX_RESULTS = int(os.environ.get("ARXIV_MAX_RESULTS", "3"))
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")

print("Loading Whisper model:", WHISPER_MODEL_NAME)
whisper_model = whisper.load_model(WHISPER_MODEL_NAME)

# instantiate the LangChain wrapper for Ollama
llm = Ollama(model="llama3")
app = FastAPI()

# ---------- TOOL IMPLEMENTATIONS & REGISTRY ----------


def search_arxiv(query: str) -> str:
    """
    Perform a real arXiv search and return a short, readable summary.

    Uses the `arxiv` Python package, which wraps the official arXiv API.
    """
    query = (query or "").strip()
    if not query:
        return "Error: arXiv query is empty."

    try:
        search = arxiv.Search(
            query=query,
            max_results=ARXIV_MAX_RESULTS,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = list(search.results())
    except Exception as e:
        return f"Error querying arXiv for '{query}': {e}"

    if not results:
        return f"No arXiv results found for '{query}'."

    parts = []
    for i, paper in enumerate(results, start=1):
        title = paper.title.strip().replace("\n", " ")
        authors = ", ".join(a.name for a in paper.authors)
        # arxiv library: summary is the abstract
        summary = (paper.summary or "").replace("\n", " ").strip()
        summary_snippet = summary[:600]
        if len(summary) > 600:
            summary_snippet += "..."

        published_date = paper.published.date() if paper.published else "unknown date"
        url = paper.entry_id  # canonical arXiv URL

        parts.append(
            f"[{i}] {title} ({published_date})\n"
            f"Authors: {authors}\n"
            f"Summary: {summary_snippet}\n"
            f"Link: {url}"
        )

    return (
        f"Top arXiv results for '{query}':\n\n" + "\n\n".join(parts)
    )


def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.
    Uses sympy first; falls back to a very restricted eval.
    Returns result as a string.
    """
    try:
        expr = sp.sympify(expression)
        result = expr.evalf()
        return str(result)
    except Exception:
        try:
            # Very restricted eval environment
            safe_globals = {
                "__builtins__": {},
                "sqrt": math.sqrt,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "pi": math.pi,
                "e": math.e,
            }
            result = eval(expression, safe_globals, {})
            return str(result)
        except Exception as e:
            return f"Error evaluating expression '{expression}': {e}"


def weather_query(city: str) -> str:
    """
    Query current weather using the OpenWeatherMap API.

    Requires the env var OPENWEATHER_API_KEY to be set.
    """
    city = (city or "").strip()
    if not city:
        return "Error: city name is empty."

    api_key = OPENWEATHER_API_KEY
    if not api_key:
        return (
            "Weather service is not configured: missing OPENWEATHER_API_KEY "
            "environment variable."
        )

    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",  # Celsius
    }

    try:
        resp = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"Error calling weather API for '{city}': {e}"

    data = resp.json()

    # OpenWeatherMap uses `cod` for status in the JSON body as well
    if str(data.get("cod")) != "200":
        msg = data.get("message", "unknown error")
        return f"Weather API error for '{city}': {msg}"

    name = data.get("name", city)
    main = data.get("main", {})
    weather_list = data.get("weather", [])
    wind = data.get("wind", {})

    temp = main.get("temp")
    feels_like = main.get("feels_like")
    humidity = main.get("humidity")
    description = weather_list[0].get("description") if weather_list else "unknown"
    wind_speed = wind.get("speed")

    # Build a nice human-readable summary
    return (
        f"Current weather in {name}: {description}, {temp}°C "
        f"(feels like {feels_like}°C), humidity {humidity}%, "
        f"wind speed {wind_speed} m/s."
    )


TOOL_REGISTRY = {
    "search_arxiv": search_arxiv,
    "calculate": calculate,
    "weather_query": weather_query,
}


SYSTEM_TOOL_PROMPT = """
You are a helpful AI assistant that has access to tools.

Available tools:

1. search_arxiv(query: str) -> str
   Use this to look up specific scientific articles or recent research.
   WARNING: The arXiv API is rate-limited, so only call this tool if you
   truly need titles/authors/abstracts or very recent papers.

2. calculate(expression: str) -> str
   Use this to evaluate math expressions.

3. weather_query(city: str) -> str
   Use this ONLY if the user explicitly asks about weather, temperature,
   forecast, or conditions in a specific place.

WHEN you want to use a tool, respond with ONLY a single JSON object
describing the tool call, for example:
{"function": "calculate", "arguments": {"expression": "2+2"}}
{"function": "search_arxiv", "arguments": {"query": "recent quantum entanglement research"}}
{"function": "weather_query", "arguments": {"city": "New York"}}

Do NOT call weather_query for questions about physics, math, history, etc.
Do NOT call search_arxiv or weather_query repeatedly if they return an error.
If a tool returns an error (for example HTTP 401 or 429), explain the problem
to the user in plain text and do NOT call that same tool again.

If you can answer the user’s question well using your own knowledge WITHOUT
calling any tools, prefer to answer directly in plain text.

You MAY call tools multiple times in a row if needed. When you see
a tool result in the conversation, you can either:
- call another tool (again with a JSON object), or
- answer in plain text.

Always try to produce a clear final answer in plain text once you have
enough information from tools.
""".strip()


def try_parse_tool_call(text: str):
    """
    Try to parse the LLM output as a tool-call JSON object:
        {"function": "...", "arguments": {...}}
    Returns (function_name, arguments_dict) on success, or None on failure.
    """
    candidate = text.strip()

    # Strip Markdown code fences if present
    if candidate.startswith("```"):
        # e.g. ```json\n{...}\n```
        candidate = candidate.strip("`")
        # remove an optional leading "json" or "JSON"
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].strip()

    # Try to isolate the outermost {...}
    if "{" in candidate and "}" in candidate:
        candidate = candidate[candidate.find("{") : candidate.rfind("}") + 1]

    try:
        obj = json.loads(candidate)
    except Exception:
        return None

    if not isinstance(obj, dict):
        return None

    fn_name = obj.get("function")
    args = obj.get("arguments")

    if not isinstance(fn_name, str) or not isinstance(args, dict):
        return None

    return fn_name, args


def call_llm(prompt: str) -> str:
    """Simple wrapper around the Ollama LLM."""
    return str(llm.invoke(prompt))


async def run_agent_with_tools(user_query: str, max_tool_loops: int = 3):
    """
    Core logic:
      - Ask LLM (with tool instructions).
      - If it returns a JSON tool call, run the tool.
      - Feed the tool result back to the LLM as context.
      - Repeat up to max_tool_loops times.
      - Return final natural-language answer.

    Returns:
      final_answer (str),
      first_raw_llm (str),
      tool_calls_log (list of dicts)
    """
    tool_messages = []  # textual descriptions of past tool calls/results
    first_raw_llm = None
    tool_calls_log = []

    for step in range(max_tool_loops + 1):
        prompt_parts = [SYSTEM_TOOL_PROMPT, f"User: {user_query}"]
        prompt_parts.extend(tool_messages)
        prompt_parts.append("Assistant:")
        full_prompt = "\n\n".join(prompt_parts)

        raw = call_llm(full_prompt)
         # Save the very first raw response for debug/logging purposes
        if first_raw_llm is None:
            first_raw_llm = raw

        parsed = try_parse_tool_call(raw)
        if not parsed:
            # Not a valid tool call -> treat as final answer
            return raw, first_raw_llm, tool_calls_log

        fn_name, args = parsed
        tool_fn = TOOL_REGISTRY.get(fn_name)

        if tool_fn is None:
            # Unknown function name -> fallback to returning the raw LLM text
            tool_calls_log.append(
                {
                    "step": step,
                    "error": f"Unknown tool '{fn_name}'",
                    "raw_llm": raw,
                }
            )
            return raw, first_raw_llm, tool_calls_log

        # Call the tool
        try:
            tool_output = tool_fn(**args)
        except TypeError as e:
            # Argument mismatch
            tool_output = f"Error calling tool '{fn_name}': {e}"
        except Exception as e:
            tool_output = f"Tool '{fn_name}' raised an error: {e}"

        error_keywords = ["Error calling", "HTTP 401", "Unauthorized", "rate-limiting", "HTTP 429"]
        if any(k in tool_output for k in error_keywords):
            # Add the tool error as context and then *force* a final natural-language answer
            tool_messages.append(
                f'Tool "{fn_name}" returned an error: {tool_output}\n'
                "Assistant: Please explain this problem to the user in plain text and "
                "do not attempt to call that tool again."
            )
            # Single extra LLM call to explain the error
            prompt_parts = [SYSTEM_TOOL_PROMPT, f"User: {user_query}"] + tool_messages
            prompt_parts.append("Assistant:")
            full_prompt = "\n\n".join(prompt_parts)
            raw = call_llm(full_prompt)
            # Return that as final
            if first_raw_llm is None:
                first_raw_llm = raw
            tool_calls_log.append(
                {
                    "step": step,
                    "function": fn_name,
                    "arguments": args,
                    "output": tool_output,
                    "note": "Stopped after tool error.",
                }
            )
            return raw, first_raw_llm, tool_calls_log

        tool_calls_log.append(
            {
                "step": step,
                "function": fn_name,
                "arguments": args,
                "output": tool_output,
            }
        )

        # Add tool result as context for the next LLM call (chained calls)
        tool_messages.append(
            f'Tool "{fn_name}" was called with arguments {json.dumps(args)} '
            f'and returned:\n{tool_output}'
        )

    # If we got here, the loop hit its limit
    fallback_msg = (
        "I’m sorry, I tried using tools multiple times and got stuck. "
        "Here is the last available information:\n\n"
    )
    if tool_calls_log:
        last = tool_calls_log[-1]
        fallback_msg += f"{last}"
    return fallback_msg, first_raw_llm, tool_calls_log


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
    path = save_temp_file(data, suffix='.webm')
    try:
        transcription = await transcribe_file(path)
    finally:
        cleanup_file(path)

    session_id = ensure_session(session_id)
    session_memories[session_id].append({'role': 'User', 'text': transcription})

    # --- NEW: use the tool-enabled agent instead of simple streaming ---
    final_answer, raw_llm, tool_calls = await run_agent_with_tools(transcription)

    session_memories[session_id].append({'role': 'Assistant', 'text': final_answer})

    # You could also return debug info if desired
    return JSONResponse({
        'session_id': session_id,
        'transcription': transcription,
        'response': final_answer,
        'raw_llm': raw_llm,
        'tool_calls': tool_calls,
    })


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
        msg = await websocket.receive_text()
        msg_json = json.loads(msg)
        speaker_id = msg_json.get('speaker_id', 'default')

        prompt = msg_json.get('prompt')
        audio_blob_b64 = msg_json.get('audio_b64')

        if audio_blob_b64 and not prompt:
            data = base64.b64decode(audio_blob_b64)
            path = save_temp_file(data, suffix='.webm')
            try:
                prompt = await transcribe_file(path)
            finally:
                cleanup_file(path)

        if not prompt:
            await websocket.send_json({'type': 'error', 'message': 'no prompt provided'})
            return

        session_memories[session_id].append({'role': 'User', 'text': prompt})
        await websocket.send_json({'type': 'prompt', 'message': prompt})

        # --- NEW: tool-enabled agent call ---
        try:
            final_answer, raw_llm, tool_calls = await run_agent_with_tools(prompt)
            response = final_answer

            # For debugging on the client, you can send the raw + tool info as well:
            await websocket.send_json({
                'type': 'raw_llm',
                'message': raw_llm,
                'tool_calls': tool_calls,
            })

            await websocket.send_json({'type': 'response', 'message': response})
            session_memories[session_id].append({'role': 'Assistant', 'text': response})

        except Exception as e:
            await websocket.send_json({'type': 'error', 'message': str(e)})
            return f"⚠️ Calling Llama3 error: {e}"

        # TTS as before
        await websocket.send_json({'type': 'tts_start'})
        await synthesize_tts_and_stream(websocket, response)
        await websocket.send_json({'type': 'tts_end'})

    except WebSocketDisconnect:
        print('client disconnected')
    except Exception as e:
        await websocket.send_json({'type': 'error', 'message': str(e)})
        raise
    
if __name__ == '__main__':
    uvicorn.run('backend.app:app', host='0.0.0.0', port=8000, reload=True)