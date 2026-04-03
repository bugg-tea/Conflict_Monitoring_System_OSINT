import requests
from bs4 import BeautifulSoup
import feedparser
import json
import time
import re

from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# CONFIG
# ==============================

ACTORS = [
    "iran", "iranian", "us", "usa", "united states"
    "pentagon", "tehran", "netanyahu"
]

EVENT_KEYWORDS = [
    "war", "attack", "strike", "missile", "drone", "bomb",
    "military", "explosion", "conflict", "retaliation",
    "escalation", "nuclear"
]

NEGATIVE_TOPICS = [
    "football", "cricket", "nba", "music", "celebrity",
    "movie", "fashion", "stock", "crypto"
]

RSS_SOURCES = {
    "CNN World":           "http://rss.cnn.com/rss/edition_world.rss",
    "CNN Middle East":     "http://rss.cnn.com/rss/edition_meast.rss",
    "BBC World":           "http://feeds.bbci.co.uk/news/world/rss.xml",
    "BBC Middle East":     "http://feeds.bbci.co.uk/news/world/middle_east/rss.xml",
    "Reuters":             "https://feeds.reuters.com/reuters/topNews",
    "Al Jazeera":          "https://www.aljazeera.com/xml/rss/all.xml",
    "Google Iran US":      "https://news.google.com/rss/search?q=iran+united+states+war+conflict&hl=en-US&gl=US&ceid=US:en",
    "Google Iran Israel":  "https://news.google.com/rss/search?q=iran+israel+attack+strike+2025&hl=en-US&gl=US&ceid=US:en",
    "Google Iran Nuclear": "https://news.google.com/rss/search?q=iran+nuclear+us+israel+2025&hl=en-US&gl=US&ceid=US:en",
    "Google Houthi":       "https://news.google.com/rss/search?q=houthi+iran+us+israel+attack&hl=en-US&gl=US&ceid=US:en",
    "Google IRGC":         "https://news.google.com/rss/search?q=irgc+iran+pentagon+strike&hl=en-US&gl=US&ceid=US:en",

}

OUTPUT_FILE = "step1_output.json"

# ==============================
# MODEL
# ==============================

model = SentenceTransformer("all-MiniLM-L6-v2")

QUERY = model.encode([
    "US Iran war",
    "US Iran conflict"
    "Iran attack Israel",
    "Middle East war escalation",
    "US military attack on Iran nuclear facilities",
    "Iran retaliates against Israel missile strike",
    "US Iran conflict sanctions military operation",
    "Iran nuclear deal uranium enrichment threat",
    "Pentagon CENTCOM Iran war strategy",
    "Trump Iran deal negotiation military option",
])

# ==============================
# FILTERS (IMPROVED)
# ==============================

def has_actor(text):
    text = text.lower()
    return any(a in text for a in ACTORS)

def has_event(text):
    text = text.lower()
    return any(e in text for e in EVENT_KEYWORDS)

def is_negative(text):
    text = text.lower()
    return any(n in text for n in NEGATIVE_TOPICS)

def semantic_score(text):
    emb = model.encode(text[:512])
    sims = cosine_similarity([emb], QUERY)[0]
    return max(sims)

# Relaxed filtering (important fix)
def is_relevant(title, text):
    combined = (title + " " + text).lower()

    if not has_actor(combined):
        return False

    if is_negative(combined):
        return False

    if not has_event(combined):
        return False

    if semantic_score(combined) < 0.35:
        return False

    return True

# ==============================
# SCRAPER
# ==============================

def fetch_article(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        title = soup.find("h1")
        title = title.get_text() if title else ""

        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text() for p in paragraphs)

        return title.strip(), text.strip()

    except Exception as e:
        print("Fetch error:", e)
        return "", ""

# ==============================
# RSS SCRAPER
# ==============================

def scrape_rss():
    articles = {}
    print("\n📡 Fetching RSS...")

    for source, url in RSS_SOURCES.items():
        try:
            feed = feedparser.parse(url)

            for entry in feed.entries:
                link = entry.get("link")

                if not link or link in articles:
                    continue

                title = entry.get("title", "")
                summary = entry.get("summary", "")

                articles[link] = {
                    "title": title,
                    "summary": summary,
                    "source": source
                }

        except Exception as e:
            print("RSS error:", e)

    print(f"Total links: {len(articles)}")
    return articles

# ==============================
# BETTER DEDUP (FIXED)
# ==============================

def deduplicate(articles, threshold=0.90):
    seen = []
    unique = []

    for a in articles:
        text = a["title"] + " " + a["text"]
        emb = model.encode(text)

        duplicate = False
        for s in seen:
            sim = cosine_similarity([emb], [s])[0][0]
            if sim > threshold:
                duplicate = True
                break

        if not duplicate:
            seen.append(emb)
            unique.append(a)

    return unique

# ==============================
# MAIN PIPELINE
# ==============================

def run_pipeline():
    rss = scrape_rss()

    results = []
    kept = 0
    dropped = 0

    print("\n🔍 Filtering...")

    for url, meta in rss.items():
        title = meta["title"]
        summary = meta["summary"]

        page_title, page_text = fetch_article(url)

        final_title = page_title if page_title else title
        final_text  = page_text if page_text else summary

        if not final_text:
            continue

        if is_relevant(final_title, final_text):
            results.append({
                "title": final_title,
                "url": url,
                "text": final_text[:800],
                "source": meta["source"]
            })
            kept += 1
        else:
            dropped += 1

        time.sleep(0.2)

    print(f"\n📊 Kept: {kept} | Dropped: {dropped}")

    print("\n🧹 Deduplicating...")
    results = deduplicate(results)

    print(f"After dedup: {len(results)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_pipeline()