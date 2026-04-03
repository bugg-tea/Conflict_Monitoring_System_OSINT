import json
import re
import time
import base64
from datetime import datetime
from dateutil import parser as dateparser
from langdetect import detect
import requests
from bs4 import BeautifulSoup
import pytz

# =========================
# CONFIG
# =========================

INPUT_FILE  = "step4_output.json"   # output of File 3
OUTPUT_FILE = "step5_output.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive",
}

# =========================
# GOOGLE NEWS URL DECODER
# =========================

def decode_google_news_url(google_url):
    """
    Decode Google News CBMi... encoded URLs to real article URL.
    Google encodes the destination URL in base64 in the path segment.
    """
    try:
        match = re.search(r'articles/([^?&]+)', google_url)
        if not match:
            return None
        encoded = match.group(1)
        # Pad to valid base64 length
        padded  = encoded + '=' * (4 - len(encoded) % 4)
        decoded = base64.urlsafe_b64decode(padded).decode('latin-1')
        # Real URL is embedded after binary prefix — find first http
        m = re.search(r'https?://[^\x00-\x1f\x7f]+', decoded)
        if m:
            real = re.sub(r'[\x00-\x1f\x7f\s]+$', '', m.group(0))
            return real
    except Exception:
        pass
    return None


def resolve_url(url):
    """
    Resolve any URL to its final destination.
    Handles Google News redirect URLs and normal HTTP redirects.
    Returns the real URL or original if resolution fails.
    """
    if not url:
        return url

    if 'news.google.com' in url:
        # Try base64 decode first (fast, no HTTP)
        decoded = decode_google_news_url(url)
        if decoded and 'google.com' not in decoded:
            return decoded
        # Fallback: follow HTTP redirect chain
        try:
            r = requests.get(url, headers=HEADERS, timeout=15,
                             allow_redirects=True)
            if 'google.com' not in r.url:
                return r.url
        except Exception:
            pass
        return url  # genuinely unresolvable Google URL

    # Normal URL — just follow redirects
    try:
        r = requests.get(url, headers=HEADERS, timeout=10,
                         allow_redirects=True)
        return r.url
    except Exception:
        return url

# =========================
# LANGUAGE DETECTION
# =========================

def detect_language(text):
    try:
        if text and len(text.strip()) > 20:
            return detect(text[:2000])
    except Exception:
        pass
    return "unknown"

# =========================
# UTC CONVERSION
# =========================

def convert_to_utc(dt):
    if not dt:
        return None
    try:
        if dt.tzinfo is not None:
            return dt.astimezone(pytz.UTC)
        return pytz.UTC.localize(dt)
    except Exception:
        return None

# =========================
# DATE PARSING
# =========================

def parse_datetime(raw):
    """Safely parse any datetime string."""
    if not raw:
        return None
    try:
        return dateparser.parse(str(raw))
    except Exception:
        return None


def extract_date_time(dt):
    """Split a datetime into ISO date string and time string."""
    if not dt:
        return None, None
    try:
        return dt.date().isoformat(), dt.strftime("%H:%M:%S")
    except Exception:
        return None, None


def try_rss_date(item):
    """
    Extract date from the 'published' field saved by File 1 RSS parser.
    This is the fastest and most reliable source — no HTTP needed.
    """
    raw = (item.get("published")
           or item.get("published_at")
           or item.get("rss_published")
           or "")
    if not raw:
        return None
    dt = parse_datetime(raw)
    return convert_to_utc(dt) if dt else None


def try_url_date(url):
    """
    Extract date embedded in the URL path, e.g. /2025/06/15/article-title.
    Many news sites include the publish date in the URL structure.
    """
    if not url:
        return None
    # Match patterns like /2024/06/15/ or /2024-06-15/
    m = re.search(
        r'[/\-_](\d{4})[/\-_](\d{2})[/\-_](\d{2})[/\-_]',
        url
    )
    if m:
        try:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            # Sanity check
            if 2020 <= y <= 2030 and 1 <= mo <= 12 and 1 <= d <= 31:
                dt = datetime(y, mo, d, tzinfo=pytz.UTC)
                return dt
        except Exception:
            pass
    return None


def extract_published_from_meta(soup):
    """Extract publish date from common HTML meta tags."""
    selectors = [
        "meta[property='article:published_time']",
        "meta[property='og:published_time']",
        "meta[name='pubdate']",
        "meta[name='publish-date']",
        "meta[name='date']",
        "meta[itemprop='datePublished']",
        "meta[name='DC.date.issued']",
        "meta[name='sailthru.date']",
        "meta[name='article.published']",
    ]
    for sel in selectors:
        tag = soup.select_one(sel)
        if tag and tag.get("content"):
            return tag["content"]
    return None


def extract_published_from_jsonld(soup):
    """Extract publish date from JSON-LD structured data blocks."""
    scripts = soup.find_all("script", type="application/ld+json")
    for script in scripts:
        if not script.string:
            continue
        try:
            data = json.loads(script.string)
        except Exception:
            continue
        # Handle both list and dict forms
        if isinstance(data, list):
            data = data[0] if data else {}
        if isinstance(data, dict):
            for field in ("datePublished", "dateCreated", "uploadDate"):
                if field in data:
                    return data[field]
    return None


def extract_published_from_time_tag(soup):
    """Extract date from <time> HTML tags."""
    for time_tag in soup.find_all("time"):
        dt_val = time_tag.get("datetime") or time_tag.get("content")
        if dt_val:
            return dt_val
        # Try inner text as last resort
        inner = time_tag.get_text(strip=True)
        if inner and len(inner) > 5:
            return inner
    return None


def fetch_page_soup(url, timeout=12):
    """Lightweight requests-based page fetch — no Playwright needed for dates."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout,
                         allow_redirects=True)
        if r.status_code == 200:
            return BeautifulSoup(r.text, "html.parser"), r.url
    except Exception as e:
        print(f"    [FETCH ERROR] {e}")
    return None, url


def extract_published_datetime(item, real_url):
    """
    Multi-strategy date extraction pipeline.
    Tries sources in order from fastest/cheapest to slowest.

    Strategy order:
      1. RSS feed date (saved in File 1, zero HTTP cost)
      2. URL path date pattern (zero HTTP cost)
      3. Fetch real article page → meta tags → JSON-LD → <time> tag
    """
    # --- Strategy 1: RSS date ---
    dt = try_rss_date(item)
    if dt:
        print(f"    [DATE] Found via RSS field")
        return dt

    # --- Strategy 2: URL path date ---
    dt = try_url_date(real_url)
    if dt:
        print(f"    [DATE] Found via URL pattern")
        return dt

    # --- Strategy 3: Parse real article page ---
    if not real_url or 'google.com' in real_url:
        return None

    print(f"    [DATE] Fetching page for date...")
    soup, _ = fetch_page_soup(real_url)
    if not soup:
        return None

    raw = (
        extract_published_from_meta(soup)
        or extract_published_from_jsonld(soup)
        or extract_published_from_time_tag(soup)
    )

    if raw:
        dt = parse_datetime(raw)
        return convert_to_utc(dt) if dt else None

    return None

# =========================
# LAST MODIFIED EXTRACTION
# =========================

def extract_last_modified_from_meta(soup):
    selectors = [
        "meta[property='article:modified_time']",
        "meta[property='og:updated_time']",
        "meta[name='last-modified']",
        "meta[itemprop='dateModified']",
    ]
    for sel in selectors:
        tag = soup.select_one(sel)
        if tag and tag.get("content"):
            return tag["content"]
    return None


def extract_last_modified_from_jsonld(soup):
    scripts = soup.find_all("script", type="application/ld+json")
    for script in scripts:
        if not script.string:
            continue
        try:
            data = json.loads(script.string)
        except Exception:
            continue
        if isinstance(data, list):
            data = data[0] if data else {}
        if isinstance(data, dict) and "dateModified" in data:
            return data["dateModified"]
    return None


def extract_last_modified_from_headers(url):
    try:
        r = requests.head(url, headers=HEADERS, timeout=8,
                          allow_redirects=True)
        lm = r.headers.get("Last-Modified") or r.headers.get("last-modified")
        if lm:
            return lm
    except Exception:
        pass
    return None


def extract_modified_datetime(real_url, soup, fallback_dt=None):
    """Extract last-modified date using multiple strategies."""
    if soup:
        raw = (
            extract_last_modified_from_meta(soup)
            or extract_last_modified_from_jsonld(soup)
        )
        if raw:
            dt = parse_datetime(raw)
            return convert_to_utc(dt) if dt else fallback_dt

    if real_url and 'google.com' not in real_url:
        raw = extract_last_modified_from_headers(real_url)
        if raw:
            dt = parse_datetime(raw)
            return convert_to_utc(dt) if dt else fallback_dt

    return fallback_dt

# =========================
# TEXT CLEANUP
# =========================

def final_text_cleanup(text):
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

# =========================
# IMAGE CLEANUP
# =========================

def clean_images(images):
    cleaned = []
    for img in (images or []):
        if not img:
            continue
        cleaned.append({
            "image_url":     img.get("image_url") or img.get("url", ""),
            "image_caption": (img.get("image_caption")
                              or img.get("caption") or ""),
            "image_text":    img.get("ocr_text") or img.get("image_text", ""),
        })
    return cleaned

# =========================
# MAIN PROCESSING
# =========================

def process(input_file, output_file):

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results  = []
    total    = len(data)
    dated    = 0
    no_date  = 0

    print(f"\n{'='*50}")
    print(f"  FILE 4 — Structured Output Pipeline")
    print(f"  Total articles: {total}")
    print(f"{'='*50}\n")

    for idx, item in enumerate(data, 1):

        # ── Identify URLs ─────────────────────────────────
        raw_url  = (item.get("url")
                    or item.get("final_url")
                    or item.get("original_url") or "")
        title    = item.get("title", "")
        text     = (item.get("clean_text")
                    or item.get("article_text") or "").strip()

        print(f"\n[{idx}/{total}] {title[:70]}")
        print(f"  URL: {raw_url[:70]}")

        # ── Resolve real URL (fix Google News redirects) ──
        real_url = resolve_url(raw_url)
        if real_url != raw_url:
            print(f"  → Resolved: {real_url[:70]}")

        # ── Language detection ────────────────────────────
        # Use title as fallback when text is very short
        lang_input = text if len(text) > 50 else (title + " " + text)
        language   = detect_language(lang_input)

        # ── Date extraction ───────────────────────────────
        pub_dt = extract_published_datetime(item, real_url)

        # Last-modified needs the soup we may have already fetched
        # Fetch page once and reuse for both pub + modified if needed
        soup = None
        if pub_dt is None and real_url and 'google.com' not in real_url:
            # We already tried once in extract_published_datetime,
            # fetch again only if we need modified too
            pass

        # For modified date, re-fetch once if we have a real URL
        if real_url and 'google.com' not in real_url:
            try:
                soup, _ = fetch_page_soup(real_url, timeout=10)
            except Exception:
                soup = None
            # Second chance for pub date via soup
            if pub_dt is None and soup:
                raw = (
                    extract_published_from_meta(soup)
                    or extract_published_from_jsonld(soup)
                    or extract_published_from_time_tag(soup)
                )
                if raw:
                    dt = parse_datetime(raw)
                    pub_dt = convert_to_utc(dt) if dt else None

        mod_dt = extract_modified_datetime(real_url, soup,
                                           fallback_dt=pub_dt)

        # ── Split into date + time components ─────────────
        published_date, published_time = extract_date_time(pub_dt)
        modified_date,  modified_time  = extract_date_time(mod_dt)

        if pub_dt:
            dated += 1
            print(f"  ✅ Published: {published_date} {published_time}")
        else:
            no_date += 1
            print(f"  ⚠️  No date found")

        # ── Image cleanup ─────────────────────────────────
        cleaned_images = clean_images(item.get("images", []))

        # ── Build output record ───────────────────────────
        new_item = item.copy()
        new_item.update({
            # URLs
            "url":                raw_url,
            "real_url":           real_url,

            # Text
            "article_text":       final_text_cleanup(text),
            "language":           language,

            # Published timestamp
            "published_at":       str(pub_dt) if pub_dt else None,
            "published_date":     published_date,
            "published_time":     published_time,

            # Last modified timestamp
            "last_modified_at":   str(mod_dt) if mod_dt else None,
            "last_modified_date": modified_date,
            "last_modified_time": modified_time,

            # Images
            "images":             cleaned_images,
        })

        results.append(new_item)

        # Polite delay to avoid rate limiting
        time.sleep(0.3)

    # ── Save output ───────────────────────────────────────
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"  PIPELINE STATS")
    print(f"{'='*50}")
    print(f"  Total processed : {total}")
    print(f"  ✅ Dates found   : {dated}")
    print(f"  ⚠️  No date       : {no_date}")
    print(f"  📦 Output saved  : {output_file}")
    print(f"{'='*50}\n")

# =========================
# RUN
# =========================

if __name__ == "__main__":
    process(INPUT_FILE, OUTPUT_FILE)