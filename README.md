# 🚀 Intelligence System (OSINT Event Extraction & Analysis)

An end-to-end intelligence monitoring system that ingests multi-source data, processes it through NLP pipelines, extracts structured events, clusters them, and presents insights through an interactive dashboard.

---

## 📌 Project Structure

```
system2.0/
│
├── app/                      # Core backend pipeline
│   ├── ingestion/           # Data collection (fetcher, link extractor)
│   ├── llm/                 # LLM utilities
│   ├── models/              # Embeddings & ML models
│   ├── pipeline/            # Pipeline orchestration
│   ├── processing/          # Core NLP + processing modules
│   ├── utils/               # Helper utilities
│   └── main.py              # Entry point
│
├── app2/                    # Dashboard (Flask + UI)
│   ├── static/              # CSS / assets
│   ├── templates/           # HTML dashboard
│   └── app.py               # Dashboard backend
│
├── models/                  # Model artifacts
│
├── analyzed_clusters.json   # Final clustered output (used by dashboard)
├── clustered_events.json
├── chunks.json
├── final_output.json
├── output.json
├── embeddings.npz
├── debug_raw.json
├── failed.json
│
└── test.py
```

---

## 🧠 System Overview

This project builds a multi-stage intelligence pipeline:

```
Ingestion → Extraction → Cleaning → NLP → Event Extraction → 
Entity Extraction → Fusion → Clustering → Analysis → Dashboard
```

---

## ⚙️ Key Components

### 🔹 1. Ingestion Layer
Fetches data from news sources

**Modules:**
- `fetcher.py`
- `link_extractor.py`

---

### 🔹 2. Processing Pipeline (`app/processing/`)

Handles all core NLP and transformation:

#### Text Cleaning
- `cleaner.py`
- `clean.py`

#### Chunking
- `chunker.py`
- `chunk_pipeline.py`

#### Event Extraction
- `event_extractor.py`
- `event_builder.py`

#### Entity Extraction
- `entity_pipeline.py`

#### Image Analysis
- `image_analysis.py` (OCR + captioning attempts)

#### Fusion
- `fusion.py`

#### Clustering
- `clustering.py`

#### Analysis & Summarization
- `analysis.py`
- `event_summarizer.py`

#### Deduplication
- `dedup.py`

---

### 🔹 3. Embedding & Models
- `models/embeddings.py`
- `embedding_pipeline.py`

Stores embeddings in:
```
embeddings.npz
```

---

### 🔹 4. Pipeline Orchestration
- `run_pipeline.py`

Coordinates full system execution.

---

### 🔹 5. Output Data

Generated structured outputs:

- `final_output.json`
- `output.json`
- `chunks.json`
- `clustered_events.json`
- `analyzed_clusters.json` (used by dashboard)

---

### 🔹 6. Dashboard (`app2/`)

Built using:
- Flask
- HTML + CSS + JS
- Plotly

**Features:**
- 📊 Trend visualization  
- 📰 Event feed  
- 🔍 Drill-down per cluster  

---

## 🚀 How to Run

### 1. Create virtual environment

```
python -m venv venv
venv\Scripts\activate
```

---

### 2. Install dependencies

```
pip install -r requirements.txt
```

---

### 3. Run pipeline (generate outputs)

```
python app/pipeline/run_pipeline.py
```

---

### 4. Start dashboard

```
cd app2
python app.py
```

---

### 5. Open in browser

```
http://127.0.0.1:5000/
```

---

## 📊 Dashboard Preview

Add screenshots inside `/images` folder

```
![Dashboard](images/dashboard.png)
![Drill Down](images/drilldown.png)
```

---

## ⚠️ Known Limitations

- ❌ Event extraction is not fully accurate (limited multi-event detection)  
- ❌ OCR + image captioning not reliable  
- ❌ LLM + NER hybrid approach not implemented due to constraints  
- ❌ Text extraction contains noise and needs improvement  
- ❌ Advanced AI features (RAG, conflict detection) not yet implemented  
- ❌ Dashboard drill-down may depend on data structure  

---

## 🔮 Future Improvements

- Improve event extraction accuracy  
- Add LLM-based summarization (RAG-based clustering insights)  
- Enhance image understanding pipeline  
- Implement conflict detection and scoring  
- Improve text cleaning and filtering  
- Build advanced intelligence reasoning layer  

---

## 👨‍💻 Author

**Purva Jivani**

---

## 📌 Notes

This project demonstrates:

- End-to-end system design  
- Real-world data pipeline engineering  
- NLP + AI integration  
- Practical problem-solving under constraints  
