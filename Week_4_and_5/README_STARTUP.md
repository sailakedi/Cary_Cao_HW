# Environment
export WHISPER_MODEL=medium
# Local Python
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
# Docker
docker-compose up --build
# Run
1. Start up backend
cd ~/Project-Chatbot/backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

2. Start up frontend
# Open another command line window, and run the following instructions:
cd ~/Project-Chatbot/frontend
npm install
npm run dev

3. Start up interface from browser
http://localhost:5174

# The Vite server will run at http://localhost:5174 . It proxies API calls and WebSocket connections to your backend ( http://localhost:8000 ). You can record audio, send text prompts, receive streaming tokens, and listen to streamed TTS audio responses. 
# Use the provided frontend/App.jsx inside a React app (Vite or CRA). Make sure
# the browser serves the frontend on same host:port or enable CORS/proxy.