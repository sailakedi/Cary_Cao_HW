#utils.py
import os
import uuid
from collections import defaultdict, deque
from typing import Deque, Dict

MAX_TURNS = 5
session_memories: Dict[str, Deque[dict]] = defaultdict(lambda:
deque(maxlen=MAX_TURNS))

def ensure_session(session_id: str = None) -> str:
    if not session_id:
       session_id = uuid.uuid4().hex
    return session_id

def save_temp_file(data: bytes, suffix: str = '.wav') -> str:
    path = f"/tmp/{uuid.uuid4().hex}{suffix}"
    with open(path, 'wb') as f:
         f.write(data)
    return path

def cleanup_file(path: str):
    try:
        os.remove(path)
    except Exception:
        pass

