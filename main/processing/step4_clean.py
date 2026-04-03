import json
import re
import unicodedata
import requests
import random

from bs4 import BeautifulSoup
from urllib.parse import urljoin
from newspaper import Article
from playwright.sync_api import sync_playwright
from table_text2 import extract_images_with_ocr
import trafilatura
from readability import Document
from sentence_transformers import SentenceTransformer, util
from ste2b import resolve_url, best_text

# Load once globally (important for performance)
model = SentenceTransformer('all-MiniLM-L6-v2')

# =========================
# CONFIG
# =========================

USER_AGENTS = [
    "Mozilla/5.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (X11; Linux x86_64)",
]

def get_headers():
    return {"User-Agent": random.choice(USER_AGENTS)}

import base64
import re

def decode_google_news_url(google_url):
    """
    Decode Google News encoded URLs to get the actual article URL.
    Google News encodes the real URL in the CBMi... part using base64.
    """
    try:
        # Extract the encoded part after 'articles/'
        match = re.search(r'articles/([^?]+)', google_url)
        if not match:
            return google_url
        
        encoded = match.group(1)
        
        # Pad base64 if needed
        padded = encoded + '=' * (4 - len(encoded) % 4)
        decoded_bytes = base64.urlsafe_b64decode(padded)
        
        # The real URL is embedded after some binary prefix — find 'http'
        decoded_str = decoded_bytes.decode('latin-1')
        url_match = re.search(r'https?://[^\x00-\x1f\x7f]+', decoded_str)
        if url_match:
            real_url = url_match.group(0).strip()
            # Remove trailing garbage characters
            real_url = re.sub(r'[\x00-\x1f\x7f\s]+$', '', real_url)
            return real_url
    except Exception as e:
        pass
    
    # Fallback: use requests to follow redirect chain
    try:
        import requests
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        # Use HEAD first, if that fails use GET
        resp = requests.get(google_url, headers=headers, 
                           timeout=15, allow_redirects=True)
        final = resp.url
        if 'google.com' not in final:
            return final
    except:
        pass
    
    return google_url  # Return original if all fails


def resolve_url(url):
    """Resolve any URL — handles Google News and regular redirects."""
    if not url:
        return url
    if 'news.google.com' in url:
        return decode_google_news_url(url)
    # For regular URLs, follow redirects
    try:
        import requests
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
        return r.url
    except:
        return url

def rank_top_images_by_title(title, images, top_k=5):
    if not images or not title:
        return images[:top_k]

    try:
        title_embedding = model.encode(title, convert_to_tensor=True)

        scored_images = []

        for img in images:
            # Combine caption + OCR text for better meaning
            img_text = (img.get("caption", "") + " " + img.get("ocr_text", "")).strip()

            if not img_text:
                continue

            img_embedding = model.encode(img_text, convert_to_tensor=True)

            similarity = util.cos_sim(title_embedding, img_embedding).item()

            scored_images.append((similarity, img))

        # Sort by similarity (highest first)
        scored_images.sort(key=lambda x: x[0], reverse=True)

        # Return top K images only
        return [img for _, img in scored_images[:top_k]]

    except Exception as e:
        print(f"[IMAGE RANKING ERROR] {e}")
        return images[:top_k]

# =========================
# TEXT CLEANING UTILITIES
# =========================

def normalize_encoding(text):
    return unicodedata.normalize("NFKC", text) if text else ""


def remove_urls(text):
    return re.sub(r'http\S+|www\.\S+', '', text)


def remove_symbols(text):
    return re.sub(r'[^\w\s.,\-:]', '', text)


def remove_garbage_lines(text):
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        line = line.strip()

        if len(line) < 20:
            continue

        if re.search(r'(subscribe|sign up|advertisement|cookie|privacy)', line.lower()):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


def fix_spacing(text):
    return re.sub(r'\s+', ' ', text).strip()


def sentence_split(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def clean_article_text(text):
    text = normalize_encoding(text)
    text = remove_urls(text)
    text = remove_symbols(text)
    text = remove_garbage_lines(text)
    text = fix_spacing(text)

    sentences = sentence_split(text)

    seen = set()
    final = []

    for s in sentences:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            final.append(s)

    return " ".join(final)


# =========================
# 🔥 SCORING SYSTEM (KEY UPGRADE)
# =========================

def score_text(text):
    if not text:
        return 0

    score = 0

    # length
    score += min(len(text) / 2000, 1.0)

    # structure
    sentences = re.split(r'[.!?]', text)
    score += min(len(sentences) / 20, 1.0)

    # penalties
    if re.search(r'(cookie|subscribe|advertisement)', text.lower()):
        score -= 0.5

    return score


# =========================
# 🔥 MULTI EXTRACTION ENGINE
# =========================

def extract_with_newspaper(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.title, article.text
    except:
        return "", ""


def extract_with_trafilatura(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded)
        return "", text if text else ""
    except:
        return "", ""


def extract_with_readability(url):
    try:
        r = requests.get(url, headers=get_headers(), timeout=10, allow_redirects=True)
        doc = Document(r.text)

        soup = BeautifulSoup(doc.summary(), "html.parser")
        text = soup.get_text(separator=" ")

        return doc.title(), text
    except:
        return "", ""

def extract_with_playwright(url):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()

            page.set_extra_http_headers(get_headers())

            page.goto(url, timeout=30000)

            # =========================
            # 🔥 STEP 1: HANDLE POPUPS
            # =========================

            try:
                # Accept cookies / close popups
                buttons = page.locator("button")

                for i in range(min(10, buttons.count())):
                    text = buttons.nth(i).inner_text().lower()

                    if any(k in text for k in [
                        "accept", "agree", "allow",
                        "close", "got it", "continue"
                    ]):
                        buttons.nth(i).click()
                        break
            except:
                pass

            # =========================
            # 🔥 STEP 2: REMOVE OVERLAYS
            # =========================

            page.evaluate("""
                () => {
                    let elements = document.querySelectorAll('div, section');

                    elements.forEach(el => {
                        let style = window.getComputedStyle(el);

                        if (
                            style.position === 'fixed' &&
                            parseInt(style.zIndex) > 1000
                        ) {
                            el.remove();
                        }
                    });
                }
            """)

            # =========================
            # 🔥 STEP 3: FORCE SCROLL
            # =========================

            page.evaluate("""
                window.scrollTo(0, document.body.scrollHeight);
            """)

            page.wait_for_timeout(3000)

            # =========================
            # 🔥 STEP 4: GET CLEAN HTML
            # =========================

            html = page.content()

            browser.close()

        soup = BeautifulSoup(html, "html.parser")

        title = soup.find("h1")
        title = title.get_text(strip=True) if title else ""
        
        article = soup.find("article") or soup.find("main") or soup

        paragraphs = article.find_all("p")

        text = " ".join(p.get_text() for p in paragraphs)

        return title, text

    except Exception as e:
        print(f"[PLAYWRIGHT ERROR] {url} -> {e}")
        return "", ""


def is_video_page(url, html=None):
    try:
        if html is None:
            html = requests.get(url, headers=get_headers(), timeout=10, allow_redirects=True).text

        soup = BeautifulSoup(html, "html.parser")

        # Check video tags
        if soup.find("video"):
            return True

        # Check iframe (YouTube, embeds)
        if soup.find("iframe"):
            return True

        # URL patterns
        if any(k in url.lower() for k in ["video", "watch", "live"]):
            return True

        return False
    except:
        return False
def smart_extract(url, article_dict=None):
    candidates = []

    methods = [
        extract_with_newspaper,
        extract_with_trafilatura,
        extract_with_readability,
        extract_with_playwright,
    ]

    for method in methods:
        try:
            title, text = method(url)
            if text and len(text) > 100:
                candidates.append({
                    "title": title,
                    "text": text,
                    "score": score_text(text)
                })
        except:
            continue

    # Last resort: use existing snippet from prior pipeline stage
    if article_dict:
        snippet = (article_dict.get("text") 
                   or article_dict.get("final_text") or "")
        if snippet and len(snippet) > 50:
            candidates.append({
                "title": article_dict.get("title", ""),
                "text": snippet,
                "score": score_text(snippet) * 0.5   # lower priority
            })

    if not candidates:
        return None

    return max(candidates, key=lambda x: x["score"])

# =========================
# 🔥 ARTICLE EXTRACTION (FINAL)
# =========================

def extract_article(url):
    result = smart_extract(url)

    if not result:
        print("[FAILED] No extraction worked")
        return None

    return result


# =========================
# 🔥 IMAGE FILTERING (IMPROVED)
# =========================

def is_valid_image_url(url):
    if not url:
        return False

    url = url.lower()

    junk_keywords = [
        "logo", "icon", "sprite", "ads", "banner",
        "placeholder", "pixel", "avatar", "thumbnail"
    ]

    if any(k in url for k in junk_keywords):
        return False

    return url.endswith((".jpg", ".jpeg", ".png", ".webp"))


def is_war_relevant(img_url, caption=""):
    keywords = [
        "iran", "us", "usa", "military", "war", "attack",
        "missile", "strike", "explosion", "drone", "tank",
        "soldier", "army", "conflict", "bomb"
    ]

    text = (img_url + " " + caption).lower()

    return any(k in text for k in keywords)


def get_article_container(soup):
    selectors = [
        "article",
        "div[itemprop='articleBody']",
        "main"
    ]

    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            return el

    return soup





# =========================
# 🔥 IMAGE EXTRACTION
# =========================


# =========================
# 🔥 MAIN PIPELINE
# =========================

def process_file(input_file, output_file):
    total_count = 0
    success_count = 0
    failed_count = 0
    video_count = 0

    with open(input_file, "r", encoding="utf-8") as f:
        articles = json.load(f)

    results = []
    
    for article in articles:

        total_count += 1

        
        raw_url = (article.get("real_url") 
                  or article.get("url") 
                  or article.get("final_url"))
        url = resolve_url(raw_url) or raw_url
        print(f"[RESOLVED] {url[:80] if url else 'NONE'}")
        
        url = resolve_url(url)

        print(f"\n[PROCESSING] {url}")

        if not url:
            failed_count += 1
            results.append({
        "url": None,
        "title": article.get("title", ""),
        "source": article.get("source", ""),
        "clean_text": "",
        "tables": article.get("tables", []),
        "images": [],
        "language": "en",
        "content_type": "article",
        "extraction_status": "failed_no_url"
    })
            continue
           

    # =========================
    # STEP 1: FETCH HTML ONCE
    # =========================
        try:
            html = requests.get(url, headers=get_headers(), timeout=10, allow_redirects=True).text
        except Exception as e:
            print(f"[HTML FETCH ERROR] {url} -> {e}")
            html = None
       

    # =========================
    # STEP 2: VIDEO DETECTION
    # =========================
        if html and is_video_page(url, html):
            print("[INFO] Video page detected")

            video_count += 1

            results.append({
                "url": article.get("original_url") or article.get("final_url"),
                "title": article.get("title") or article.get("final_title"),
                "source": article.get("source"),
                "clean_text": "",
                "tables": article.get("tables", []),
                "images": [],
                "language": "en",
                "content_type": "video",
                "extraction_status": "skipped_video"
        })

            continue

    # =========================
    # STEP 3: EXTRACTION
    # =========================
        
        extracted = smart_extract(url, article_dict=article)

    # =========================
    # STEP 4: FALLBACK LOGIC 🔥
    # =========================
    
        # STEP 4: FALLBACK LOGIC — IMPROVED
        if not extracted or not extracted.get("text"):
            print("[WARNING] Extraction failed → trying best_text fallback")

            fallback_text = best_text(url) if url else ""  # <-- try fresh extraction

            if not fallback_text:
                fallback_text = (
                    article.get("text")
                    or article.get("final_text")
                    or ""
        )

            cleaned_text = clean_article_text(fallback_text) if fallback_text else ""

            status = "text_extracted_fallback" if len(cleaned_text) > 100 else "fallback_used"
            failed_count += 1

            results.append({
        "url": url,
        "title": article.get("title") or article.get("final_title"),
        "source": article.get("source"),
        "clean_text": cleaned_text,
        "tables": article.get("tables", []),
        "images": [],
        "language": "en",
        "content_type": "article",
        "extraction_status": status
    })
            continue
        
    # =========================
    # STEP 5: SUCCESS CASE
    # =========================
        success_count += 1
 
        cleaned_text = clean_article_text(extracted["text"])

        try:
            soup = BeautifulSoup(html, "html.parser") if html else None
            all_images = extract_images_with_ocr(soup, url) if soup else []

            title = article.get("title") or article.get("final_title") or extracted.get("title")

            images = rank_top_images_by_title(title, all_images, top_k=5)
            
            
        except:
            images = []

        results.append({
            "url": article.get("original_url") or article.get("final_url"),
            "title": article.get("title") or article.get("final_title"),
            "source": article.get("source"),
            "clean_text": cleaned_text,
            "tables": article.get("tables", []),
            "images": images,
            "language": "en",
            "content_type": "article",
            "extraction_status": "success"
    })

   
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print("\n==========================")
    print("📊 PIPELINE STATS")
    print("==========================")
    print(f"Total Articles     : {total_count}")
    print(f"✅ Successful       : {success_count}")
    print(f"⚠️ Fallback Used    : {failed_count}")
    print(f"🎥 Video Skipped    : {video_count}")
    print(f"📦 Output Articles  : {len(results)}")
    print("==========================\n") 

    print(f"\n✅ Saved to {output_file}")


# =========================
# RUN
# =========================

if __name__ == "__main__":
    process_file(
        input_file="step2_output.json",
        output_file="step4_output.json"
    )