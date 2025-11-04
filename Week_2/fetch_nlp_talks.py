import os
os.environ["XDG_CACHE_HOME"] = os.path.expanduser("~/MyProjects/Week_2/.cache")  # pick any folder you own

import json
import subprocess
import pytesseract
import ffmpeg
import yt_dlp
from PIL import Image
from tqdm import tqdm
import whisper

# ---------- CONFIG ----------
QUERY = "NLP conference talk"
MAX_VIDEOS = 10
FRAME_INTERVAL = 30          # seconds between screenshots
OUTPUT_DIR = "nlp_talks"
OUTPUT_JSONL = "nlp_talks.jsonl"
MAX_DURATION = 2400           # 40 min max duration
# ----------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize Whisper model for speech transcription
whisper_model = whisper.load_model("base")

def search_youtube(query, max_results=10):
    """Use yt-dlp to search for YouTube videos."""
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'extract_flat': True,
        'default_search': 'ytsearch',
        'noplaylist': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
        return result['entries']

def download_audio_and_frames(video_url, out_dir, frame_interval=30):
    """Download audio and extract frames from YouTube video."""
    os.makedirs(out_dir, exist_ok=True)
    ydl_opts = {
        'outtmpl': f'{out_dir}/%(title)s.%(ext)s',
        'format': 'bestaudio/best',
        'quiet': True,
    }

    # Download audio file 
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        audio_path = os.path.join(out_dir, f"{info['title']}.webm")
        video_path = audio_path
        return info, audio_path, video_path

def extract_frames(video_path, frame_dir, interval=30):
    """Extract frames every N seconds using ffmpeg."""
    os.makedirs(frame_dir, exist_ok=True)
    try:
        (
            ffmpeg
            .input(video_path)
            .filter('fps', fps=1/interval)
            .output(os.path.join(frame_dir, 'frame_%04d.jpg'), vframes=None)
            .run(quiet=True, overwrite_output=True)
        )
        frames = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith('.jpg')])
        return frames
    except Exception as e:
        print("Frame extraction failed:", e)
        return []

def ocr_frames(frames):
    """Run OCR with Tesseract on frames and concatenate results."""
    texts = []
    for f in frames:
        try:
            img = Image.open(f)
            text = pytesseract.image_to_string(img)
            if text.strip():
                texts.append(text.strip())
        except Exception as e:
            print("OCR failed for", f, ":", e)
    return "\n".join(texts)

def transcribe_audio(audio_path):
    """Transcribe audio using Whisper."""
    try:
        result = whisper_model.transcribe(audio_path)
        return result['text']
    except Exception as e:
        print("Whisper failed:", e)
        return ""

def main():
    videos = search_youtube(QUERY, MAX_VIDEOS)
    print(f"Found {len(videos)} videos for query '{QUERY}'.")

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out_f:
        for v in tqdm(videos, desc="Processing videos"):
            title = v.get('title')
            url = v.get('url')
            duration = v.get('duration') or 0

            # Skip if too long
            if duration > MAX_DURATION:
                continue

            print(f"Downloading: {title} ({duration}s)")

            safe_title = "".join(c if c.isalnum() else "_" for c in title)[:50]
            vid_dir = os.path.join(OUTPUT_DIR, safe_title)
            os.makedirs(vid_dir, exist_ok=True)

            info, audio_path, video_path = download_audio_and_frames(url, vid_dir, FRAME_INTERVAL)

            # Extract frames every FRAME_INTERVAL seconds
            frames = extract_frames(video_path, os.path.join(vid_dir, "frames"), FRAME_INTERVAL)

            # OCR text from frames
            ocr_text = ocr_frames(frames)

            # Transcribe speech
            transcript = transcribe_audio(audio_path)

            # Save entry as JSONL
            entry = {
                "title": title,
                "url": url,
                "duration": duration,
                "timestamp": info.get("upload_date"),
                "ocr_text": ocr_text,
                "transcript": transcript
            }
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ Saved results to {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
