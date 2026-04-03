# step2b_reextract.py
# Run this on iran_us_conflict_enriched.json BEFORE running file 3
# This fixes the empty text problem by resolving real URLs first

import json
import time
import re
import base64
import requests
from bs4 import BeautifulSoup
import trafilatura
from newspaper import Article
from readability import Document

INPUT_FILE  = "step2_output.json"
OUTPUT_FILE = "step3_output.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# ─────────────────────────────────────────
# URL RESOLUTION
# ─────────────────────────────────────────

def decode_google_news_url(google_url):
    try:
        match = re.search(r'articles/([^?&]+)', google_url)
        if not match:
            return None
        encoded = match.group(1)
        padded  = encoded + '=' * (4 - len(encoded) % 4)
        decoded = base64.urlsafe_b64decode(padded).decode('latin-1')
        m = re.search(r'https?://[^\x00-\x1f\x7f]+', decoded)
        if m:
            return re.sub(r'[\x00-\x1f\x7f\s]+$', '', m.group(0))
    except:
        pass
    return None


def resolve_url(url):
    if not url:
        return None
    if 'news.google.com' in url:
        decoded = decode_google_news_url(url)
        if decoded and 'google.com' not in decoded:
            return decoded
        # fallback: follow redirect with requests
        try:
            r = requests.get(url, headers=HEADERS, timeout=15,
                             allow_redirects=True)
            if 'google.com' not in r.url:
                return r.url
        except:
            pass
        return None          # genuinely unresolvable
    try:
        r = requests.get(url, headers=HEADERS, timeout=10,
                         allow_redirects=True)
        return r.url
    except:
        return url


# ─────────────────────────────────────────
# TEXT EXTRACTION  (3 methods + scoring)
# ─────────────────────────────────────────

def score_text(text):
    if not text:
        return 0
    words = len(text.split())
    penalty = len(re.findall(r'(subscribe|cookie|advertisement|sign up)',
                             text.lower()))
    return words - penalty * 20


def extract_trafilatura(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            no_fallback=False
        )
        return text or ""
    except:
        return ""


def extract_newspaper(url):
    try:
        a = Article(url)
        a.download()
        a.parse()
        return a.text or ""
    except:
        return ""


def extract_readability(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=12,
                         allow_redirects=True)
        doc  = Document(r.text)
        soup = BeautifulSoup(doc.summary(), "html.parser")
        return soup.get_text(separator=" ").strip()
    except:
        return ""


def best_text(url):
    """Try all extractors, return highest-scoring result."""
    candidates = []
    for fn in [extract_trafilatura, extract_newspaper, extract_readability]:
        try:
            t = fn(url)
            if t:
                candidates.append(t)
        except:
            pass
    if not candidates:
        return ""
    return max(candidates, key=score_text)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def run():
    with open(INPUT_FILE, encoding="utf-8") as f:
        articles = json.load(f)

    results = []
    for art in articles:
        original_url = (art.get("original_url")
                        or art.get("final_url")
                        or art.get("url") or "")

        print(f"\n[URL] {original_url[:80]}")

        real_url = resolve_url(original_url)
        print(f" → resolved: {real_url and real_url[:80]}")

        existing_text = art.get("text", "").strip()

        # Only re-extract if text is short / empty
        if real_url and len(existing_text) < 200:
            new_text = best_text(real_url)
        else:
            new_text = ""

        final_text = new_text if len(new_text) > len(existing_text) else existing_text

        art["real_url"]    = real_url or original_url
        art["text"]        = final_text
        art["text_length"] = len(final_text)

        results.append(art)
        time.sleep(0.3)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    filled   = sum(1 for r in results if len(r.get("text","")) > 100)
    print(f"\n✅ Saved {OUTPUT_FILE}")
    print(f"   Articles with text > 100 chars : {filled}/{len(results)}")


if __name__ == "__main__":
    run()