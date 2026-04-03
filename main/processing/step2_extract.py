import requests
import json
import re
import pandas as pd
import easyocr

from bs4 import BeautifulSoup
from urllib.parse import urljoin
from io import BytesIO
from PIL import Image

from playwright.sync_api import sync_playwright



# -----------------------------
# CONFIG
# -----------------------------
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

reader = easyocr.Reader(['en'], gpu=False)

# -----------------------------
# STEP 1: RESOLVE REDIRECT (CRITICAL FIX)
# -----------------------------
def resolve_final_url(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10, allow_redirects=True)
        return response.url
    except Exception as e:
        print(f"[REDIRECT ERROR] {url} -> {e}")
        return url

# -----------------------------
# STEP 2: FETCH HTML (PLAYWRIGHT SAFE)
# -----------------------------
def fetch_rendered_html(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
)
        page = context.new_page()

        try:
            page.goto(url, wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(3000)
            html = page.content()

        except Exception as e:
            print(f"[PLAYWRIGHT ERROR] {url} -> {e}")
            html = None

        finally:
            browser.close()

    return html

# -----------------------------
# STEP 3: CLEAN TEXT
# -----------------------------
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# -----------------------------
# STEP 4: EXTRACT MAIN TEXT
# -----------------------------
def extract_main_text(soup):
    selectors = [
        "article",
        "main",
        "div[class*='content']",
        "div[class*='article']",
        "div[id*='content']",
    ]

    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            return clean_text(el.get_text(separator=" "))

    return clean_text(soup.get_text(separator=" "))

# -----------------------------
# STEP 5: TABLE EXTRACTION
# -----------------------------
def extract_tables(html):
    tables_data = []

    try:
        dfs = pd.read_html(html)
        for i, df in enumerate(dfs):
            tables_data.append({
                "table_index": i,
                "data": df.to_dict(orient="records")
            })
    except Exception:
        pass

    return tables_data

# -----------------------------
# STEP 6: IMAGE FILTER (REMOVE ADS / ICONS)
# -----------------------------
def is_valid_image(url):
    if not url:
        return False

    url = url.lower()

    junk_keywords = [
        "logo", "icon", "sprite", "ads", "banner",
        "placeholder", "pixel", "avatar"
    ]

    if any(k in url for k in junk_keywords):
        return False

    return url.endswith((".jpg", ".jpeg", ".png", ".webp"))

# -----------------------------
# STEP 7: EXTRACT IMAGE CAPTION
# -----------------------------
def get_image_caption(img_tag):
    # Common caption patterns
    caption = ""

    if img_tag.has_attr("alt"):
        caption = img_tag["alt"]

    # Look for nearby caption elements
    parent = img_tag.find_parent()
    if parent:
        cap_tag = parent.find("figcaption")
        if cap_tag:
            caption = cap_tag.get_text(strip=True)

    return caption.strip()

# -----------------------------
# STEP 8: EXTRACT IMAGES + OCR + CAPTION
# -----------------------------
def extract_images_with_ocr(soup, base_url):
    images_output = []
    img_urls = set()

    # 1. IMG TAGS
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-original")

        if src:
            img_urls.add(urljoin(base_url, src))

        # srcset handling
        if img.get("srcset"):
            parts = [p.split(" ")[0] for p in img["srcset"].split(",")]
            for p in parts:
                img_urls.add(urljoin(base_url, p))

    # 2. FILTER + PROCESS
    for img_url in img_urls:
        if not is_valid_image(img_url):
            continue

        try:
            response = requests.get(img_url, headers=HEADERS, timeout=10)

            if len(response.content) < 5000:
                continue

            image = Image.open(BytesIO(response.content)).convert("RGB")

            # OCR
            ocr_text = " ".join(reader.readtext(response.content, detail=0))

            # Find caption from soup (best-effort)
            img_tag = soup.find("img", src=lambda x: x and img_url.split("/")[-1] in x if x else False)
            caption = get_image_caption(img_tag) if img_tag else ""

            images_output.append({
                "image_url": img_url,
                "caption": caption,
                "ocr_text": ocr_text
            })

        except Exception:
            continue

    return images_output

# -----------------------------
# STEP 9: PROCESS SINGLE ARTICLE
# -----------------------------
def process_article(article):
    original_url = article["url"]
    final_url = resolve_final_url(original_url)
    html = fetch_rendered_html(final_url)

    if not html:
        # ✅ Don't return None — return a degraded record instead
        return {
            "title": article.get("title", ""),
            "original_url": original_url,
            "final_url": final_url,
            "source": article.get("source", ""),
            "text": article.get("text", ""),   # fallback to RSS text
            "tables": [],
            "images": [],
            "extraction_status": "failed_no_html"
        }

    soup = BeautifulSoup(html, "html.parser")
    text = extract_main_text(soup)
    tables = extract_tables(html)
    images = extract_images_with_ocr(soup, final_url)

    return {
        "title": article.get("title", ""),
        "original_url": original_url,
        "final_url": final_url,
        "source": article.get("source", ""),
        "text": text or article.get("text", ""),  # fallback too
        "tables": tables,
        "images": images,
        "extraction_status": "success"
    }

# -----------------------------
# STEP 10: RUN PIPELINE
# -----------------------------
def run_pipeline(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        articles = json.load(f)

    results = []

    for article in articles:
        processed = process_article(article)
        if processed:
            results.append(processed)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Saved to {output_file}")

# -----------------------------
# EXECUTION
# -----------------------------
if __name__ == "__main__":
    run_pipeline(
        input_file="step1_output.json",
        output_file="step2_output.json"
    )