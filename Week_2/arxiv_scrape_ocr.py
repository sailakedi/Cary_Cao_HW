import time
import json
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import trafilatura
import os
from pathlib import Path
from typing import List, Dict, Optional
import logging

#Playwright and OCR imports (optional fallback)
from PIL import Image
import pytesseract

#Import Playwright inside try block so script can still run if Playwright isn't available.
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except Exception:
    PLAYWRIGHT_AVAILABLE = False

# ---------------- Config ----------------------
ARXIV_API_BASE = "http://export.arxiv.org/api/query"
USER_AGENT = "arXiv-scraper/1.0 (python requests)"
REQUESTS_SLEEP = 0.1 # polite delay between sequests to arXiv
SCREENSHOT_FOLDER = "abs_screenshots"
OUTPUT_JSON = "arxiv_papers.json"
TIMEOUT = 20 # requests timeout

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(messge)s")

def fetch_arxiv_ids(category: str, max_results: int = 200) -> List[str]:
    """
    Use arXiv API to fetch latest paper ids for a given category.
    Returns a list of arXiv IDs (e.g., '2309.12345').
    """
    #arXiv API query: search_query=cat:cs.CL&sortBy=submittedDate&sortOrder=descending
    params = {
        "search_query": f"cat:{category}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "srtOrder": "descending"
    }
    headers = {"User-Agent": USER_AGENT}
    response = requests.get(ARXIV_API_BASE, params=params, headers=headers, timeout=TIMEOUT)
    response.raise_for_status()
    xml = response.text

    # parse xml for <id> tags that contain the arXiv id URL: entries have <id>http://arxiv.org/abs/IDvN</id>
    soup = BeautifulSoup(xml, "xml")
    entries = soup.find_all("entry")
    ids = []
    for entry in entries:
        # extract arXiv id from id tag or id-raw tag
        id_tag = entry.find("id")
        if id_tag and id_tag.text:
            #id like "http://arxiv.org/abs/2309.12345v1"
            url = id_tag.text.strip()
            #extract id portion after last '/'
            arxiv_id = url.rsplit("/",1)[-1]
            # remove version if desired (keep version? we'll remove for canonical)
            arxiv_id_no_ver = arxiv_id.split("v")[0]
            ids.append(arxiv_id_no_ver)

    logging.info("Fetched %d arXiv ids from API category %s", len(ids), category)
    return ids

def Fetch_abs_html(abs_url: str) -> Optional[str]:
    """
    Fetch the HTML content of an arXiv abstract page.
    Uses Playwright to render JavaScript if available, otherwise falls back to requests.
    Returns the HTML content as a string, or None on failure.
    """
    headers = {"User-Agent": USER_AGENT}
    try:
        r = requests.get(abs_url, headers=headers, timeout=TIMEOUT)
        r.raise_for_status()
        return r.text
    except Exception as e:
        logging.warning("Failed to fetch %s : %s", abs_url, e)
        return None
    
def parse_abs_page(html: str, url: str) -> Dict:
    """
    Parse title, authors, date from arXiv /abs/ HTML using BeautifulSoup and meta tags.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Title: prefer meta citation_title
    title = None
    meta_title = soup.find("meta", attrs={"name": "citation_title"})
    if meta_title and meta_title.get("content"):
        title = meta_title["content"].strip()
    else:
        #fallback: h1 title element
        h1 = soup.find("h1", class_="title") or soup.find("h1")
        if h1:
            #h1 may contain "Title: ..." text; remove "Title:"
            title = h1.get_text(strip=True).replace("Title:", "").strip()

    # Authors: arXiv uses meta citation_author multiple times
    authors = []
    for meta in soup.find_all("meta", attrs={"name": "citation_author"}):
        if meta.get("content"):
            authors.append(meta["content"].strip())
    if not authors:
        #fallback: parse authors block
        auth_block = soup.find("div", class_="authors")
        if auth_block:
            # authors are in <a> tags
            authors = [a.get_text(strip=True) for a in auth_block.find_all("a")]
            if not authors:
                # try spliting text
                txt = auth_block.get_text(separator=",").replace("Authors:","").strip()
                authors = [a.strip() for a in txt.split(",") if a.strip()]

    # Date: try citation_date or submission history
    date = None
    meta_date = soup.find("meta", attrs={"name": "citation_date"})
    if meta_date and meta_date.get("cntent"):
        date = meta_date["content"].strip()
    else:
        # fallback: find submission history first line
        sub_hist = soup.find("div", class_="submission-history")
        if sub_hist:
            first_line = sub_hist.get_text().strip().split("\n")[0]
            date = first_line.strip()

    # Abstract: try to extract the element explicity
    abstract_text = ""
    ab_div = soup.find("blockquote", class_="abstract")
    if ab_div:
        # blockquote's text often begins with "Abstract: " - remove that
        abstract_text = ab_div.get_text(separator="\n", strip=True).replace("Abstract:", "",1).strip()

    return {
        "url": url,
        "title": title or "",
        "authors": authors,
        "date": date or "",
        "abstract_html": abstract_text or ""
    }               

def trafilatura_extract(html: str) -> str:
    """ 
    Use trafilatura to extract page text. We'll try to find a reasonable abstract-like snippet
    """
    downloaded = trafilatura.extract(html, include_comments=False, include_tables=False, include_formatting=False, favor_precision=True)
    if not downloaded:
        return ""
    # trafilatura returns full page text: for arXiv we want an abstract-sized chunk.
    # Heuristic: look for "Abstract" marker in returned text and slice a short region after it.
    text = downloaded.strip()
    lower = text.lower()
    if "abstract" in lower:
        idx = lower.find("abstract")
        # return next -2000 chars after marker
        return text[idx: idx + 2000].strip()
    # fallback: return first 2000 chars
    return text[:20000]

def screenshot_page(url: str, screenshot_path: str, viewport={"width": 1200, "height":1600}, timeout=15000) -> bool:
    """
    Take a full-page screenshot of a URL using Playwright.
    Return True if saved successfully, False otherwise.
    """
    if not PLAYWRIGHT_AVAILABLE:
        logging.warning("Playwright not available, cannot take screenshot of %s", url)
        return False

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport=viewport)
            page = context.new_page()
            page.goto(url, timeout=timeout)
            page.wait_for_selector("body", timeout=timeout)
            page.screenshot(path=screenshot_path, full_page=True)
            browser.close()
        logging.info("Saved screenshot to %s", screenshot_path)
        return True
    except Exception as e:
        logging.warning("Playwright screenshot failed for %s : %s", url, e)
        return False
    
def ocr_image_to_text(image_path: str, lang: str = "eng") -> str:
    """
    Use pytesseract to perform OCR on an image and return extracted text.
    """
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang=lang)
        return text.strip()
    except Exception as e:
        logging.warning("TesseractOCR failed for image %s : %s", image_path, e)
        return ""
    
def ensure_folder(path: str):
    """
    Ensure that a folder exists at the specified path.
    """
    try:
        Path(path).mkdir(pparents=True, exist_ok=True)
    except Exception as e:
        logging.warning("Failed to create folder %s : %s", path, e)

def main(category: str = "cs.CL", max_papers: int = 200, output_json: str = OUTPUT_JSON):
    """
    Main function to fetch arXiv papers, extract metadata, take screenshots, and save to JSON.
    """
    ids = fetch_arxiv_ids(category, max_results=max_papers)
    ensure_folder(SCREENSHOT_FOLDER)
    results = []

    for arxiv_id in tqdm(ids, desc="Processing arXiv papers"):
        abs_url = f"https://arxiv.org/abs/{arxiv_id}"
        html = Fetch_abs_html(abs_url)
        if not html:
            logging.warning("Skipping %s due to fetch failure", abs_url)
            continue
        parsed = parse_abs_page(html, abs_url)

        # First try trafilatura on the raw HTML to get a cleaned abstract / text
        trafilatura_text = trafilatura_extract(html)
        if trafilatura_text:
            # heuristic: if the ex tracted text contains 'abstract' or is reasonably short,
            # prefer the manual blockquote abstract if present.
            abstract = parsed.get("abstract_html","").strip() or trafilatura_text.strip()
        else:
            # Try the blockquote abstract if present
            abstract = parsed.get("abstract_html", "").strip()

        # if still empty or too short, fallback to screenshot + OCR
        if (not abstract) or (len(abstract) < 50):
            screenshot_path = os.path.join(SCREENSHOT_FOLDER, f"{arxiv_id}.png")
            success = screenshot_page(abs_url, screenshot_path)
            if success:
                ocr_text = ocr_image_to_text(screenshot_path, lang="eng")
                # heuristic: try to extract "Abstract" substring from OCR text
                if ocr_text:
                    lower_ocr = ocr_text.lower()
                    if "abstract" in lower_ocr:
                        idx = lower_ocr.find("abstract")
                        # get up to 20000 chars after abstract marker
                        abstract = ocr_text[idx: idx + 2000].strip()
                    else:
                        # fallback to first ~2000 chars of OCR
                        abstract = ocr_text[:2000].strip()
                else:
                     logging.warning("No OCR fallback available for %s", abs_url)


        results.append({
            "url": parsed["url"],
            "title": parsed["title"],
            "authors": parsed["authors"],
            "date": parsed["date"],
            "abstract": abstract
        })

        # polite sleep to avoid hammering
        time.sleep(REQUESTS_SLEEP)

    # Save results as JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logging.info("Saved %d papers to %s", len(results), output_json)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scrape latest arXiv papers for a subcategory and save JSON.")
    parser.add_argument("--category", type=str, default="cs.CL", help="arXiv category (default: cs.CL)")
    parser.add_argument("--max_papers", type=int, default=200, help="Maximum number of papers to fetch (default: 200)")
    parser.add_argument("--output_json", type=str, default=OUTPUT_JSON, help="Output JSON file path (default: arxiv_papers.json)")
    args = parser.parse_args()
    main(category=args.category, max_papers=args.max_papers, output_json=args.output_json)
    