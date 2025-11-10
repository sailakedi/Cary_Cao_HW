import React, { useEffect, useRef, useState } from 'react';

export default function App() {
  const [sessionId, setSessionId] = useState(() => Math.random().toString(36).slice(2));
  const [prompt, setPrompt] = useState('Hello, how are you?');
  const [speakerId, setSpeakerId] = useState('default');
  const [tokens, setTokens] = useState('');
  const [recording, setRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const wsRef = useRef(null);
  const audioCtxRef = useRef(null);
  const audioQueueRef = useRef([]);
  const playingRef = useRef(false);

  useEffect(() => {
    audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)();
    return () => {
      if (audioCtxRef.current) audioCtxRef.current.close();
    };
  }, []);

  useEffect(() => {
    let raf;
    const tick = () => {
      if (!playingRef.current && audioQueueRef.current.length > 0) {
         playNextBuffer();
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  const connect = (promptText, audioB64 = null) => {
    if (wsRef.current) wsRef.current.close();
    const ws = new WebSocket(`ws://${location.host}/ws/${sessionId}`);
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => {
      ws.send(JSON.stringify({ prompt: promptText, audio_b64: audioB64, speaker_id: speakerId }));
    };
    
    ws.onmessage = async (event) => {
      // 🟣 1. If message is binary (ArrayBuffer or Blob), play it
      if (event.data instanceof Blob || event.data instanceof ArrayBuffer) {
        console.log("Received binary audio data");
        // Normalize to Blob
        const blob =
          event.data instanceof Blob
            ? event.data
            : new Blob([event.data], { type: "audio/wav" });

        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        try {
          await audio.play();
          console.log("Audio played");
        } catch (err) {
          console.error("Audio play failed", err);
        }
        return; // stop here; don't JSON.parse binary data
      }

      // 🟢 2. Otherwise, assume JSON/text
      let msg;
      try {
        msg = JSON.parse(event.data);
      } catch (err) {
        console.warn("Non-JSON text received:", event.data);
        return;
      }

      if (msg.type === "tts_start") console.log("TTS start");
      else if (msg.type === "tts_end") console.log("TTS end");
      else if (msg.type === "prompt") console.log("Prompt is:", msg.message);
      else if (msg.type === "response") console.log("Response is:", msg.message);
      else if (msg.type === "token") setResponse((prev) => prev + msg.token);
      else if (msg.type === "error") console.error("Server error:", msg.message);
    };

    
    ws.onclose = () => console.log('WebSocket closed');
    wsRef.current = ws;
  };

  const startChat = () => {
    setTokens('');
    connect(prompt);
  };

  const toggleRecording = async () => {
    if (recording) {
      mediaRecorderRef.current.stop();
      setRecording(false);
      return;
    }

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);
    chunksRef.current = [];
    mediaRecorder.ondataavailable = (e) => chunksRef.current.push(e.data);
    mediaRecorder.onstop = async () => {
      const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64 = reader.result.split(',')[1];
        setTokens('');
        connect(null, base64);
      };
      reader.readAsDataURL(blob);
    };
    mediaRecorderRef.current = mediaRecorder;
    mediaRecorder.start();
    setRecording(true);
  };

  const playNextBuffer = () => {
    const audioCtx = audioCtxRef.current;
    const buf = audioQueueRef.current.shift();
    if (!buf) return;
    playingRef.current = true;
    const src = audioCtx.createBufferSource();
    src.buffer = buf;
    src.connect(audioCtx.destination);
    src.onended = () => {
      playingRef.current = false;
    };
    src.start();
  };

  return (
    <div style={{ padding: 20 }}>
      <h2>Realtime Chatbot (Streaming + Microphone)</h2>
        <div>
          <label>Session ID: </label>
          <input value={sessionId} onChange={(e) => setSessionId(e.target.value)} />
        </div>
        <div style={{ marginTop: 8 }}>
        <label>Speaker: </label>
          <select value={speakerId} onChange={(e) => setSpeakerId(e.target.value)}> 
            <option value="default">Default</option>
            <option value="alice">Alice</option>
            <option value="bob">Bob</option>
          </select>
        </div>
        <div style={{ marginTop: 8 }}>
           <textarea
            rows={4}
            cols={60}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
           />
        </div>
        <div style={{ marginTop: 8 }}>
          <button onClick={startChat}>Send Text Prompt</button>
          <button onClick={toggleRecording} style={{ marginLeft: 10 }}>
            {recording ? 'Stop Recording' : 'Record Audio'}
          </button>
        </div>
        <div style={{ marginTop: 20 }}>
          <h4>Assistant Response (Streaming)</h4>
          <div style={{
              whiteSpace: 'pre-wrap',
              border: '1px solid #ccc',
              padding: 10,
              minHeight: 100,
            }}>
              {tokens}
          </div>
        </div>
    </div>
  );
}
