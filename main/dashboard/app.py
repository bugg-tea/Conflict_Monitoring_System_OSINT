from flask import Flask, render_template, jsonify
import pandas as pd
import json
import os
from collections import defaultdict

app = Flask(__name__)

BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE_STATS_PATH = os.path.join(BASE_DIR, "live_stats_copy.json")
NER_PATH        = os.path.join(BASE_DIR, "final_ner4_copy.json")
CLUSTER_PATH    = os.path.join(BASE_DIR, "clustered_output4_copy.json")

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠ Could not load {path}: {e}")
        return {}

# ── BASIC ROUTES ──────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/live")
def live():
    return jsonify(load_json(LIVE_STATS_PATH))

@app.route("/api/ner")
def ner():
    data = load_json(NER_PATH)
    if isinstance(data, list):
        return jsonify(data)
    return jsonify([])

@app.route("/api/clusters")
def clusters():
    data = load_json(CLUSTER_PATH)
    if isinstance(data, list):
        return jsonify(data)
    return jsonify([])

# ── CLUSTER DETAIL ────────────────────────────────────────
@app.route("/api/cluster/<int:cluster_id>")
def cluster_detail(cluster_id):
    data = load_json(CLUSTER_PATH)
    if isinstance(data, list):
        for c in data:
            if c.get("cluster_id") == cluster_id:
                return jsonify(c)
    return jsonify({"error": "Not found"}), 404

# ── NER GRAPH DATA ────────────────────────────────────────
@app.route("/api/graph/ner")
def ner_graph():
    """Build actor-location graph from NER data"""
    data = load_json(NER_PATH)
    if not isinstance(data, list):
        return jsonify({"nodes": [], "links": []})

    nodes = {}
    links = []
    link_set = set()

    for item in data:
        if not item.get("conflict"):
            continue

        actors    = item.get("top_actors", [])[:3]
        locations = item.get("top_locations", [])[:2]
        sentiment = item.get("actor_sentiment", {})

        # Actor nodes
        for actor in actors:
            name = actor.get("name", "")
            if not name or len(name) < 2:
                continue
            if name not in nodes:
                nodes[name] = {
                    "id":        name,
                    "label":     name,
                    "type":      "actor",
                    "count":     0,
                    "sentiment": sentiment.get(name, "NEUTRAL"),
                }
            nodes[name]["count"] += 1

        # Location nodes
        for loc in locations:
            name = loc.get("name", "")
            geo  = loc.get("geo", {})
            if not name:
                continue
            if name not in nodes:
                nodes[name] = {
                    "id":    name,
                    "label": name,
                    "type":  "location",
                    "count": 0,
                    "lat":   geo.get("lat") if geo else None,
                    "lon":   geo.get("lon") if geo else None,
                }
            nodes[name]["count"] += 1

        # Links: actor ↔ actor (co-occurrence)
        actor_names = [a.get("name","") for a in actors if a.get("name","")]
        for i in range(len(actor_names)):
            for j in range(i+1, len(actor_names)):
                key = tuple(sorted([actor_names[i], actor_names[j]]))
                if key not in link_set:
                    link_set.add(key)
                    links.append({
                        "source": actor_names[i],
                        "target": actor_names[j],
                        "type":   "co_occurs",
                        "weight": 1,
                    })
                else:
                    for l in links:
                        if set([l["source"], l["target"]]) == set(key):
                            l["weight"] = l.get("weight", 1) + 1

        # Links: actor → location
        for actor in actor_names:
            for loc in locations:
                loc_name = loc.get("name", "")
                if not loc_name:
                    continue
                key = (actor, loc_name)
                if key not in link_set:
                    link_set.add(key)
                    links.append({
                        "source": actor,
                        "target": loc_name,
                        "type":   "located_in",
                        "weight": 1,
                    })

    return jsonify({
        "nodes": list(nodes.values()),
        "links": links,
    })

# ── CLUSTER GRAPH DATA ────────────────────────────────────
@app.route("/api/graph/clusters")
def cluster_graph():
    """Build cluster → actor/location graph"""
    data = load_json(CLUSTER_PATH)
    if not isinstance(data, list):
        return jsonify({"nodes": [], "links": []})

    nodes = {}
    links = []

    for cluster in data:
        cid   = cluster.get("cluster_id")
        cname = f"Cluster {cid}"
        num   = cluster.get("num_articles", 0)
        summ  = (cluster.get("summary") or "")[:120]

        nodes[cname] = {
            "id":       cname,
            "label":    cname,
            "type":     "cluster",
            "count":    num,
            "summary":  summ,
            "cluster_id": cid,
        }

        # Actor nodes from cluster
        for actor in (cluster.get("top_actors") or [])[:3]:
            if not actor or len(actor) < 2:
                continue
            if actor not in nodes:
                nodes[actor] = {"id": actor, "label": actor, "type": "actor", "count": 0}
            nodes[actor]["count"] += 1
            links.append({"source": cname, "target": actor, "type": "has_actor"})

        # Org nodes
        for org in (cluster.get("top_organizations") or [])[:3]:
            if not org or len(org) < 2:
                continue
            if org not in nodes:
                nodes[org] = {"id": org, "label": org, "type": "org", "count": 0}
            nodes[org]["count"] += 1
            links.append({"source": cname, "target": org, "type": "has_org"})

        # Location nodes
        for loc in (cluster.get("top_locations") or [])[:2]:
            if not loc:
                continue
            if loc not in nodes:
                nodes[loc] = {"id": loc, "label": loc, "type": "location", "count": 0}
            nodes[loc]["count"] += 1
            links.append({"source": cname, "target": loc, "type": "has_location"})

    return jsonify({
        "nodes": list(nodes.values()),
        "links": links,
    })

# ── LIVE VS ARTICLES COMPARISON ───────────────────────────
@app.route("/api/comparison")
def comparison():
    """Two POV: live scrape vs extracted articles"""
    live = load_json(LIVE_STATS_PATH)
    ner  = load_json(NER_PATH)

    # POV 1: Live scraped pages
    wiki_stats = live.get("wiki_stats", [])
    live_killed  = max((w.get("killed",  0) for w in wiki_stats), default=0)
    live_injured = max((w.get("injured", 0) for w in wiki_stats), default=0)
    live_attacks = max((w.get("attacks", 0) for w in wiki_stats), default=0)

    # POV 2: Extracted from articles
    article_killed  = live.get("aggregate", {}).get("total_killed",  0)
    article_injured = live.get("aggregate", {}).get("total_injured", 0)
    article_attacks = live.get("aggregate", {}).get("total_attacks", 0)

    return jsonify({
        "live_scrape": {
            "source":  "Wikipedia / Al Jazeera Live Tracker",
            "killed":  live_killed,
            "injured": live_injured,
            "attacks": live_attacks,
            "pages":   [w.get("name") for w in wiki_stats],
        },
        "articles": {
            "source":        "Extracted from scraped articles (NER pipeline)",
            "killed":        article_killed,
            "injured":       article_injured,
            "attacks":       article_attacks,
            "total_articles": len([i for i in (ner if isinstance(ner, list) else []) if i.get("conflict")]),
        }
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)