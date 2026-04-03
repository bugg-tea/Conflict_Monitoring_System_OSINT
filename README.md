# 🚀 Intelligence System (OSINT Event Extraction & Analysis)

---

## Overview
It is an end-to-end automated intelligence pipeline that monitors, extracts, enriches, and analyzes news about the Iran-US conflict in near real-time.  

It ingests articles from RSS feeds and Google News, resolves redirect URLs, extracts full article text using multiple extraction engines, performs:
- Named Entity Recognition (NER)
- Event extraction
- Semantic clustering
- Risk scoring  

Finally, it presents everything through a live web dashboard.

---

## Project Structure
```
processing/
├── step1_rss_ingest.py          # RSS feed ingestion + semantic filtering
├── step2_enrich.py              # Playwright-based full text + image extraction
├── step2b_reextract.py          # Google News URL resolution + text fallback
├── step3_clean_extract.py       # Multi-engine text extraction + cleaning
├── step4_structured_output.py   # Date extraction + language detection + structuring
├── step5_ner_pipeline.py        # NER + event extraction + relation mining
├── step6_cluster.py             # Semantic clustering + cluster summarization
├── step7_live_tracker.py        # Live stats + risk scoring + final output
dashboard/
├── app.py                   # Flask backend serving the dashboard
└── index.html               # Frontend dashboard UI
└── README.md
```

---

## Pipeline Run Order
```
step1 → step2 → step2b → step3 → step4 → step5 → step6 → step7
```

Each step:
- Reads the output JSON of the previous step  
- Writes its own output JSON  

---

## Output Files

| File                | Produced By | Description |
|---------------------|------------|-------------|
| step1_output.json   | step1      | Filtered RSS articles with title, url, text, source |
| step2_output.json   | step2      | Enriched articles with full text, images, tables |
| step3_output.json   | step3     | URL-resolved articles with best-effort text |
| step4_output.json   | step4      | Cleaned text, video detection, extraction status |
| step5_output.json   | step5      | Structured articles with dates, language, images |
| step6_output.json   | step6      | NER, actors, locations, events, relations, conflict flag |
| step7_output.json   | step7      | Semantic clusters with summaries |
| step8_output.json   | step8      | Final intelligence output with risk score and live stats |

---

## Setup

### Install Dependencies
```bash
pip install requests beautifulsoup4 feedparser sentence-transformers scikit-learn
pip install trafilatura newspaper3k readability-lxml playwright easyocr
pip install spacy geopy transformers torch langdetect python-dateutil pytz
pip install flask pandas pillow tqdm

python -m playwright install chromium
python -m spacy download en_core_web_sm
```

---

## Running the Pipeline

Run each step in order:

```bash
python step1_rss_ingest.py
python step2_enrich.py
python step2b_reextract.py
python step3_clean_extract.py
python step4_structured_output.py
python step5_ner_pipeline.py
python step6_cluster.py
python step7_live_tracker.py
```

### Run Dashboard
```bash
cd dashboard
python app.py
```

Open your browser at:
```
http://localhost:5000
```
All output data files are in Data folder, and dashboard_images folder are also attached to see  the dashboard overview images
---

## Key Design Decisions

- All models used are **free and open-source** (no paid APIs).
- Google News CBMi URLs are decoded using **base64** before attempting HTTP redirects.
- Text extraction uses a **multi-engine approach**:
  - trafilatura  
  - newspaper3k  
  - readability-lxml  
  (ranked by quality)
- Event extraction uses **spaCy dependency parsing** instead of LLMs:
  - Faster  
  - Deterministic  
  - No GPU required  
- Risk scoring:
  - 30% rule-based heuristics  
  - 70% zero-shot classification using `facebook/bart-large-mnli`

---

## Known Issues

- Some Google News URLs:
  - Cannot be base64 decoded  
  - Block redirects  
  → Result: empty text (fallback to RSS summary)

- Articles behind paywalls (e.g., WSJ, Bloomberg):
  → Partial or no text extraction  

- `flan-t5-base` summarization:
  → Produces low-quality summaries for complex clusters  

- EasyOCR:
  → Slow on CPU (can become bottleneck)  

- Date extraction:
  → Fails for ~30–40% of Google News articles  
  → Causes:
  - Paywalls  
  - 403 responses  

## 👨‍💻 Author

**Purva Jivani**

---

## 📌 Notes

This project demonstrates:

- End-to-end system design  
- Real-world data pipeline engineering  
- NLP + AI integration  
- Practical problem-solving under constraints  
