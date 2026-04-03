import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from collections import Counter
from transformers import pipeline

# =========================
# CONFIG
# =========================

INPUT_FILE = "step6_output.json"
OUTPUT_FILE = "step7_output.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

SIMILARITY_THRESHOLD = 0.50 

# ======================
# LOAD DATA
# ======================

def load_data():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# EMBEDDINGS
# =========================

def compute_embeddings(titles, model):
    return model.encode(titles, show_progress_bar=True)


# =========================
# STRICT CLUSTERING
# =========================

def cluster_articles(embeddings):
    similarity_matrix = cosine_similarity(embeddings)
    distance_matrix = 1 - similarity_matrix

    clustering = AgglomerativeClustering(
        metric='precomputed',
        linkage='complete',
        distance_threshold=1 - SIMILARITY_THRESHOLD,
        n_clusters=None
    )

    labels = clustering.fit_predict(distance_matrix)
    return labels


# =========================
# LLM SUMMARY
# =========================

summarizer = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    tokenizer="google/flan-t5-base",
    max_length=512,
    device=-1
)

def generate_summary(cluster_articles):
    
    if len(cluster_articles) == 1:
        text = cluster_articles[0].get("clean_text") or cluster_articles[0].get("article_text") or ""
        return text[:300] if text else "No content available."

    combined_text = ""

    for art in cluster_articles:
        text = art.get("clean_text") or art.get("article_text") or ""
        combined_text += text[:500] + "\n\n"

    if not combined_text.strip():
        return "No content available."

    prompt = f"""
Summarize the following news articles into a structured intelligence report.

Include:
- What happened
- Who is involved
- Where and when
- Key implications

Articles:
{combined_text}

Summary:
"""

    try:
        response = summarizer(
            prompt,
            max_length=300,
            min_length=80,
            do_sample=False
        )

        return response[0]["generated_text"].strip()

    except Exception as e:
        print("[FLAN-T5 ERROR]", e)
        return "Summary generation failed."


# =========================
# EXTRACT TOP ENTITIES
# =========================
def get_top_entities(articles, key, top_k_per_article=2):
    all_items = []

    for art in articles:
        entities = art.get("entities", {}).get(key, [])

        # take up to 2 per article (your requirement)
        if entities:
            all_items.extend(entities[:top_k_per_article])

    if not all_items:
        return []

    counter = Counter(all_items)

    # FINAL LIMIT = 2 × number of articles in cluster
    max_limit = 2 * len(articles)

    return [item for item, _ in counter.most_common(max_limit)]
# =========================
# MERGE CLUSTERS
# =========================

def build_clusters(data, labels):
    clusters = {}

    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(data[idx])

    final_clusters = []

    for cluster_id, articles in tqdm(clusters.items(), desc="Processing clusters"):

        summary = generate_summary(articles)

        # ✅ NEW: extract top entities
        top_actors = get_top_entities(articles, "people", 2)
        top_orgs = get_top_entities(articles, "organizations", 2)
        top_locations = get_top_entities(articles, "locations", 2)

        cluster_obj = {
            "cluster_id": int(cluster_id),
            "num_articles": len(articles),

            # NEW: top entities at top
            "top_actors": top_actors,
            "top_organizations": top_orgs,
            "top_locations": top_locations,

            "summary": summary,

            # Existing fields unchanged
            "urls": [a.get("url") for a in articles],
            "titles": [a.get("title") for a in articles],
            "sources": [a.get("source") for a in articles],
            "published_dates": [a.get("published_date") for a in articles],
            "published_times": [a.get("published_time") for a in articles],
            "last_modified_dates": [a.get("last_modified_date") for a in articles],

            "clean_texts": [a.get("clean_text") for a in articles],
            "article_texts": [a.get("article_text") for a in articles],
            "images": [a.get("images") for a in articles],
            "tables": [a.get("tables") for a in articles],

            "articles": articles
        }

        final_clusters.append(cluster_obj)

    return final_clusters


# =========================
# MAIN PIPELINE
# =========================

def main():
    print("Loading data...")
    data = load_data()

    titles = [d.get("title", "") for d in data]

    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Computing embeddings...")
    embeddings = compute_embeddings(titles, model)

    print("Clustering articles...")
    labels = cluster_articles(embeddings)

    print("Building clusters + summaries...")
    clusters = build_clusters(data, labels)

    print("Saving output...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)

    print("✅ DONE")


if __name__ == "__main__":
    main()