import os
import json
import re
from pathlib import Path
from html import unescape
from bs4 import BeautifulSoup
from datasketch import MinHash, MinHashLSH
from langdetect import detect, DetectorFactory
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

# make langdetect deterministic
DetectorFactory.seed = 0
nltk.download("punkt", quiet=True)

# ---------- CONFIG ----------
DATA_DIR = Path("/users/jiancao/MyProjects/Week_2")
FILES = [
    "Algorithms Fourth Edition_text.txt",
    "arxiv_papers.json",
    "nlp_talks.jsonl",
]
OUTPUT_TXT = DATA_DIR / "clean_corpus.txt"
STATS_MD = DATA_DIR / "stats.md"

SIMILARITY_THRESHOLD = 0.7
NUM_PERMUTATIONS = 128  # for MinHash
LANGUAGE_FILTER = {"en"}  # keep only English
# ----------------------------


# ---------- HELPERS ----------
def load_file(path: Path):
    """Load text or JSON/JSONL files into a list of text documents."""
    docs = []
    if path.suffix in {".json", ".jsonl"}:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    # collect all string fields that are long enough
                    # Case 1: JSON object (dictionary)
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            if isinstance(v, str) and len(v) > 50:
                                docs.append(v)
                    # Case 2: Plain string line
                    elif isinstance(obj, str) and len(obj) > 50:
                        docs.append(obj)
                    # Case 3: List of strings or dicts
                    elif isinstance(obj, list):
                        for item in obj:
                            if isinstance(item, str) and len(item) > 50:
                                docs.append(item)
                            elif isinstance(item, dict):
                                for k, v in item.items():
                                    if isinstance(v, str) and len(v) > 50:
                                        docs.append(v)
                except json.JSONDecodeError:
                    continue
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            docs.append(f.read())
    return docs


def detect_language(text: str) -> str:
    """Detect language using langdetect."""
    try:
        return detect(text)
    except Exception:
        return "unknown"


def strip_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    soup = BeautifulSoup(text, "html.parser")
    clean = soup.get_text(separator=" ")
    clean = unescape(clean)
    return re.sub(r"\s+", " ", clean).strip()


def remove_pii(text: str) -> str:
    """Remove emails, phone numbers, and credit card numbers."""
    patterns = [
        r"\b[\w\.-]+@[\w\.-]+\.\w+\b",          # emails
        r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",   # phone numbers
        r"\b(?:\d[ -]*?){13,16}\b",             # credit cards
    ]
    for pat in patterns:
        text = re.sub(pat, "[REDACTED]", text)
    return text


def remove_repetitive_ngrams(text: str, n=4):
    """Remove repetitive n-grams."""
    tokens = word_tokenize(text)
    if not tokens:
        return text
    seen = set()
    result = []
    for ng in ngrams(tokens, n):
        if ng in seen:
            continue
        seen.add(ng)
        result.append(ng[0])
    # Add last few tokens
    result.extend(tokens[-(n - 1):])
    return " ".join(result)


def get_minhash(text: str):
    """Generate a MinHash signature for text."""
    m = MinHash(num_perm=NUM_PERMUTATIONS)
    tokens = set(word_tokenize(text.lower()))
    for t in tokens:
        m.update(t.encode("utf8"))
    return m


# ---------- MAIN PIPELINE ----------
print("📂 Loading input files...")
all_texts = []
for f in FILES:
    path = DATA_DIR / f
    if path.exists():
        all_texts.extend(load_file(path))
    else:
        print(f"⚠️  File not found: {path}")

print(f"Loaded {len(all_texts)} raw documents.")

# Step 1: Language detection
print("🌐 Detecting languages...")
lang_filtered = [t for t in all_texts if detect_language(t) in LANGUAGE_FILTER]
print(f"Kept {len(lang_filtered)} {list(LANGUAGE_FILTER)} documents.")

# Step 2: Strip HTML
cleaned = [strip_html(t) for t in lang_filtered]

# Step 3: MinHash deduplication
print("🔍 Performing MinHash deduplication...")
lsh = MinHashLSH(threshold=SIMILARITY_THRESHOLD, num_perm=NUM_PERMUTATIONS)
deduped = []

for text in cleaned:
    mh = get_minhash(text)
    if not lsh.query(mh):
        key = f"doc_{len(deduped)}"
        lsh.insert(key, mh)
        deduped.append(text)

print(f"Deduplicated to {len(deduped)} unique documents.")

# Step 4: Remove PII
no_pii = [remove_pii(t) for t in deduped]

# Step 5: Remove repetitive n-grams
final_texts = [remove_repetitive_ngrams(t) for t in no_pii]

# Step 6: Save merged corpus
print("💾 Saving cleaned corpus...")
with open(OUTPUT_TXT, "w", encoding="utf-8") as out:
    for doc in final_texts:
        out.write(doc.strip() + "\n\n")

# Step 7: Compute and save stats
print("📊 Generating statistics...")
original_tokens = sum(len(word_tokenize(t)) for t in cleaned)
final_tokens = sum(len(word_tokenize(t)) for t in final_texts)
removed_pct = 100 * (1 - final_tokens / max(original_tokens, 1))

stats = f"""# Clean Corpus Statistics

- Original documents: {len(all_texts)}
- After language filter: {len(lang_filtered)}
- After deduplication: {len(final_texts)}
- Original token count: {original_tokens:,}
- Final token count: {final_tokens:,}
- Reduction: {removed_pct:.2f}%
"""

with open(STATS_MD, "w", encoding="utf-8") as f:
    f.write(stats)

print("✅ Cleaning complete!")
print(stats)
