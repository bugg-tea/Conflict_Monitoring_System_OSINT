import json
import re
import requests
import os
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from collections import defaultdict
from transformers import pipeline as hf_pipeline

# ============================================================
# CONFIG
# ============================================================

# Your existing output files
NER_FILE      = "step6_output.json"
CLUSTER_FILE  = "step7_output.json"
STRUCT_FILE   = "step5_output.json"
OUTPUT_FILE   = "step8_output.json"

os.makedirs("data", exist_ok=True)

# Live tracker pages only (already have analysis)
LIVE_PAGES = [
    {
        "url": "https://en.wikipedia.org/wiki/2026_Iran%E2%80%93United_States_war",
        "name": "Wikipedia: 2026 Iran-US War",
        "type": "wikipedia"
    },
    {
        "url": "https://www.aljazeera.com/news/2026/3/1/us-israel-attacks-on-iran-death-toll-and-injuries-live-tracker",
        "name": "Al Jazeera Live Tracker",
        "type": "aljazeera"
    },
    {
        "url": "https://en.wikipedia.org/wiki/Operation_Epic_Fury",
        "name": "Wikipedia: Operation Epic Fury",
        "type": "wikipedia"
    },
]

# ============================================================
# LOAD SMALL LOCAL LLM (zero-shot, CPU friendly)
# ============================================================
print("🧠 Loading LLM classifier...")
try:
    classifier = hf_pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1
    )
    LLM_AVAILABLE = True
    print("✅ LLM loaded")
except Exception as e:
    print(f"⚠ LLM not available: {e}")
    LLM_AVAILABLE = False

# ============================================================
# REGEX PATTERNS FOR EXTRACTION
# ============================================================

NUMBER_PATTERNS = [
    r'(\d[\d,]*)\s+(?:people\s+)?(?:killed|dead|died|fatalities)',
    r'(?:killed|dead|died)\s+(?:at least\s+)?(\d[\d,]*)',
    r'at least\s+(\d[\d,]*)\s+(?:killed|dead|died)',
    r'death toll.*?(\d[\d,]*)',
    r'(\d[\d,]*)\s+(?:were|are|have been)\s+(?:killed|dead)',
    r'killing\s+(?:at least\s+)?(\d[\d,]*)',
    r'(\d[\d,]*)\s+casualties',
]

INJURED_PATTERNS = [
    r'(\d[\d,]*)\s+(?:people\s+)?(?:wounded|injured|hurt)',
    r'(?:wounded|injured)\s+(?:at least\s+)?(\d[\d,]*)',
    r'at least\s+(\d[\d,]*)\s+(?:wounded|injured)',
    r'(\d[\d,]*)\s+(?:were|are|have been)\s+(?:wounded|injured)',
]

ATTACK_PATTERNS = [
    r'(\d[\d,]*)\s+(?:airstrikes?|strikes?|attacks?|missiles?|drones?)',
    r'(?:struck|hit|bombed|attacked)\s+(?:more than\s+)?(\d[\d,]*)\s+(?:targets?|sites?|locations?)',
    r'(\d[\d,]*)\s+(?:targets?|sites?)\s+(?:struck|hit|bombed|destroyed)',
    r'launched\s+(?:more than\s+)?(\d[\d,]*)\s+(?:missiles?|drones?|rockets?)',
]

def parse_number(s):
    try:
        return int(str(s).replace(",", "").replace(" ", ""))
    except:
        return 0

def extract_max_number(text, patterns):
    numbers = []
    text = text.lower()
    for p in patterns:
        matches = re.findall(p, text)
        for m in matches:
            n = parse_number(m)
            if 0 < n < 1000000:
                numbers.append(n)
    return max(numbers) if numbers else 0

# ============================================================
# STEP 1 — SCRAPE LIVE TRACKER PAGES ONLY
# ============================================================

def scrape_live_pages():
    results = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for page in LIVE_PAGES:
        print(f"\n📡 Scraping: {page['name']}")
        try:
            r = requests.get(page["url"], headers=headers, timeout=15)
            if r.status_code != 200:
                print(f"   ⚠ Status {r.status_code}")
                continue

            soup = BeautifulSoup(r.text, "html.parser")

            # ── Wikipedia specific ──
            if page["type"] == "wikipedia":
                # Get infobox (has structured casualty data)
                infobox_text = ""
                infobox = soup.find("table", {"class": lambda c: c and "infobox" in c})
                if infobox:
                    infobox_text = infobox.get_text(" ", strip=True)

                # Get first 5000 chars of article
                content = soup.find("div", {"id": "mw-content-text"})
                paras = content.find_all("p") if content else []
                article_text = " ".join(p.get_text() for p in paras[:30])

                full_text = infobox_text + " " + article_text

            # ── Al Jazeera live tracker specific ──
            elif page["type"] == "aljazeera":
                # Get all article content
                article = soup.find("article") or soup.find("main") or soup
                paras = article.find_all("p")
                full_text = " ".join(p.get_text() for p in paras[:50])

            else:
                paras = soup.find_all("p")
                full_text = " ".join(p.get_text() for p in paras[:30])

            if not full_text.strip():
                print(f"   ⚠ Empty content")
                continue

            # Extract numbers
            killed  = extract_max_number(full_text, NUMBER_PATTERNS)
            injured = extract_max_number(full_text, INJURED_PATTERNS)
            attacks = extract_max_number(full_text, ATTACK_PATTERNS)

            print(f"   → Killed: {killed} | Injured: {injured} | Attacks: {attacks}")
            print(f"   → Text length: {len(full_text)} chars")

            results.append({
                "source":      page["name"],
                "url":         page["url"],
                "type":        page["type"],
                "full_text":   full_text[:3000],
                "killed":      killed,
                "injured":     injured,
                "attacks":     attacks,
                "fetched_at":  datetime.now(timezone.utc).isoformat(),
            })

        except Exception as e:
            print(f"   ❌ Error: {e}")

    return results

# ============================================================
# STEP 2 — EXTRACT STATS FROM EXISTING NER FILE
# ============================================================

def extract_from_ner():
    print("\n📂 Reading existing NER data...")
    try:
        with open(NER_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"   ⚠ Could not load {NER_FILE}: {e}")
        return {}

    total_killed  = 0
    total_injured = 0
    total_attacks = 0
    actor_counts  = defaultdict(int)
    location_counts = defaultdict(int)
    event_types   = defaultdict(int)
    severity_scores = []
    sentiment_map = defaultdict(list)
    timeline      = defaultdict(list)
    source_counts = defaultdict(int)
    confirmed_count = 0
    events_with_casualties = []

    ATTACK_KEYWORDS = [
        "airstrike", "missile", "drone", "bomb", "attack",
        "strike", "explosion", "launched", "fired", "hit",
        "targeted", "destroyed", "killed", "wounded"
    ]

    for item in data:
        text = (item.get("article_text") or item.get("clean_text") or "").lower()
        title = (item.get("title") or "").lower()
        source = item.get("source", "unknown")
        pub_date = item.get("published_date", "")
        conflict = item.get("conflict", False)

        if not conflict:
            continue

        source_counts[source] += 1

        # Extract casualties from text
        killed  = extract_max_number(text, NUMBER_PATTERNS)
        injured = extract_max_number(text, INJURED_PATTERNS)
        attacks = extract_max_number(text, ATTACK_PATTERNS)

        total_killed  += killed
        total_injured += injured
        total_attacks += attacks

        # Actor counts from NER
        for actor in item.get("top_actors", []):
            name = actor.get("name", "")
            if name and len(name) > 2:
                actor_counts[name] += 1

        # Location counts
        for loc in item.get("top_locations", []):
            name = loc.get("name", "")
            if name:
                location_counts[name] += 1

        # Sentiment
        for actor, sentiment in item.get("actor_sentiment", {}).items():
            sentiment_map[actor].append(sentiment)

        # Classify event type from text
        etype = "general"
        if any(k in text for k in ["airstrike", "air strike", "bombing"]):
            etype = "airstrike"
        elif any(k in text for k in ["missile", "ballistic", "rocket"]):
            etype = "missile_attack"
        elif any(k in text for k in ["drone", "uav"]):
            etype = "drone_attack"
        elif any(k in text for k in ["nuclear", "uranium", "enrichment"]):
            etype = "nuclear"
        elif any(k in text for k in ["sanction", "diplomacy", "deal", "talks", "ceasefire"]):
            etype = "diplomatic"
        elif any(k in text for k in ["naval", "ship", "fleet", "sea", "strait"]):
            etype = "naval"
        elif any(k in text for k in ["cyber", "hack", "malware"]):
            etype = "cyber"
        event_types[etype] += 1

        # Severity (rule-based first)
        sev = 3
        if any(k in text for k in ["nuclear", "mass casualty"]): sev = 10
        elif killed > 50 or any(k in text for k in ["killed", "dead", "casualties"]): sev = 9
        elif killed > 10 or any(k in text for k in ["wounded", "explosion", "attack"]): sev = 8
        elif any(k in text for k in ["troops", "military operation"]): sev = 7
        elif any(k in text for k in ["drone", "intercept", "sanctions"]): sev = 6
        elif any(k in text for k in ["warns", "threatens"]): sev = 5
        elif any(k in text for k in ["protest"]): sev = 4
        elif any(k in text for k in ["statement", "says"]): sev = 3
        elif any(k in text for k in ["talks", "negotiation"]): sev = 2
        severity_scores.append(sev)

        # Timeline
        if pub_date:
            timeline[pub_date].append({"severity": sev, "killed": killed, "injured": injured})

        if killed > 0 or injured > 0:
            events_with_casualties.append({
                "title":   title[:100],
                "source":  source,
                "date":    pub_date,
                "killed":  killed,
                "injured": injured,
                "type":    etype,
                "url":     item.get("url", ""),
            })

    # Finalize sentiment
    final_sentiment = {}
    for actor, sentiments in sentiment_map.items():
        neg = sentiments.count("NEGATIVE")
        pos = sentiments.count("POSITIVE")
        final_sentiment[actor] = "NEGATIVE" if neg > pos else "POSITIVE"

    # Timeline aggregation
    severity_by_date = {}
    volume_by_date   = {}
    casualties_by_date = {}
    for date, entries in timeline.items():
        severity_by_date[date]   = round(sum(e["severity"] for e in entries) / len(entries), 2)
        volume_by_date[date]     = len(entries)
        casualties_by_date[date] = {
            "killed":  sum(e["killed"]  for e in entries),
            "injured": sum(e["injured"] for e in entries),
        }

    avg_severity = round(sum(severity_scores) / len(severity_scores), 2) if severity_scores else 0

    print(f"   → Conflict articles processed: {sum(source_counts.values())}")
    print(f"   → Total killed (from articles): {total_killed}")
    print(f"   → Total injured (from articles): {total_injured}")
    print(f"   → Avg severity: {avg_severity}")

    return {
        "total_killed":          total_killed,
        "total_injured":         total_injured,
        "total_attacks":         total_attacks,
        "avg_severity":          avg_severity,
        "severity_scores":       severity_scores,
        "actor_counts":          dict(sorted(actor_counts.items(), key=lambda x: -x[1])[:15]),
        "location_counts":       dict(sorted(location_counts.items(), key=lambda x: -x[1])[:10]),
        "event_type_counts":     dict(event_types),
        "actor_sentiment":       final_sentiment,
        "severity_by_date":      dict(sorted(severity_by_date.items())),
        "volume_by_date":        dict(sorted(volume_by_date.items())),
        "casualties_by_date":    dict(sorted(casualties_by_date.items())),
        "source_counts":         dict(source_counts),
        "events_with_casualties": sorted(events_with_casualties, key=lambda x: -(x["killed"]+x["injured"]))[:20],
        "total_articles":        sum(source_counts.values()),
    }

# ============================================================
# STEP 3 — EXTRACT FROM CLUSTERS
# ============================================================

def extract_from_clusters():
    print("\n📂 Reading cluster data...")
    try:
        with open(CLUSTER_FILE, "r", encoding="utf-8") as f:
            clusters = json.load(f)
    except Exception as e:
        print(f"   ⚠ Could not load {CLUSTER_FILE}: {e}")
        return []

    summaries = []
    for cluster in clusters:
        summary = cluster.get("summary", "")
        titles  = cluster.get("titles", [])
        sources = cluster.get("sources", [])
        num     = cluster.get("num_articles", 0)
        dates   = [d for d in cluster.get("published_dates", []) if d]

        if not summary or len(summary) < 50:
            continue

        summaries.append({
            "cluster_id":   cluster.get("cluster_id"),
            "num_articles": num,
            "summary":      summary[:400],
            "titles":       titles[:3],
            "sources":      list(set(s for s in sources if s)),
            "dates":        dates[:3],
        })

    print(f"   → {len(summaries)} valid clusters found")
    return summaries

# ============================================================
# STEP 4 — LLM SCORING (on top of rule-based)
# ============================================================

def llm_score_events(live_pages, ner_stats):
    """
    Use LLM to:
    1. Classify escalation level
    2. Verify risk assessment
    3. Score key live page texts
    Give 30% weight to rule-based, 70% to LLM
    """
    if not LLM_AVAILABLE:
        print("\n⚠ LLM not available, using rule-based only")
        return None, None

    print("\n🧠 Running LLM scoring...")

    # ── Escalation classification ──
    escalation_label = "MEDIUM"
    escalation_confidence = 0.5

    try:
        # Use best available text
        texts_to_classify = []
        for page in live_pages:
            if page.get("full_text"):
                texts_to_classify.append(page["full_text"][:500])

        if texts_to_classify:
            combined = " ".join(texts_to_classify)[:600]
            result = classifier(
                combined,
                candidate_labels=[
                    "critical military escalation and major casualties",
                    "high tension active conflict ongoing",
                    "moderate conflict with diplomatic activity",
                    "low tension de-escalation or ceasefire",
                ],
                multi_label=False
            )
            top = result["labels"][0]
            conf = result["scores"][0]

            if "critical" in top:
                escalation_label = "CRITICAL"
            elif "high" in top:
                escalation_label = "HIGH"
            elif "moderate" in top:
                escalation_label = "MEDIUM"
            else:
                escalation_label = "LOW"

            escalation_confidence = round(conf, 3)
            print(f"   → LLM Escalation: {escalation_label} (conf: {conf:.2f})")

    except Exception as e:
        print(f"   ⚠ LLM escalation error: {e}")

    # ── Risk label classification ──
    risk_label_llm = "MEDIUM"
    try:
        # Build context from stats
        context = f"""
        Total killed: {ner_stats.get('total_killed', 0)}.
        Total injured: {ner_stats.get('total_injured', 0)}.
        Average severity: {ner_stats.get('avg_severity', 0)}/10.
        Escalation: {escalation_label}.
        Active conflict articles: {ner_stats.get('total_articles', 0)}.
        """

        result = classifier(
            context,
            candidate_labels=[
                "critical crisis requiring immediate action",
                "high risk active conflict",
                "medium risk elevated tension",
                "low risk routine monitoring",
            ],
            multi_label=False
        )
        top = result["labels"][0]
        if "critical" in top:   risk_label_llm = "CRITICAL"
        elif "high" in top:     risk_label_llm = "HIGH"
        elif "medium" in top:   risk_label_llm = "MEDIUM"
        else:                   risk_label_llm = "LOW"

        print(f"   → LLM Risk Label: {risk_label_llm} (conf: {result['scores'][0]:.2f})")

    except Exception as e:
        print(f"   ⚠ LLM risk error: {e}")

    return {
        "escalation_label":      escalation_label,
        "escalation_confidence": escalation_confidence,
        "risk_label":            risk_label_llm,
    }, escalation_confidence

# ============================================================
# STEP 5 — COMPUTE RISK SCORE (rule + LLM merged)
# ============================================================

def compute_risk_score(ner_stats, live_pages, llm_results):
    killed   = ner_stats.get("total_killed", 0)
    injured  = ner_stats.get("total_injured", 0)
    attacks  = ner_stats.get("total_attacks", 0)
    avg_sev  = ner_stats.get("avg_severity", 0)

    # Add live page numbers (more authoritative)
    for page in live_pages:
        killed  = max(killed,  page.get("killed", 0))
        injured = max(injured, page.get("injured", 0))
        attacks = max(attacks, page.get("attacks", 0))

    # ── Rule-based score (30% weight) ──
    attack_score   = min(attacks / 50, 1.0)  * 30
    severity_score = (avg_sev / 10)          * 30
    casualty_score = min(killed / 500, 1.0)  * 25

    escalation_map = {"LOW": 0, "MEDIUM": 5, "HIGH": 10, "CRITICAL": 15}

    # Rule-based escalation
    if avg_sev >= 8:   rule_escalation = "CRITICAL"
    elif avg_sev >= 6: rule_escalation = "HIGH"
    elif avg_sev >= 4: rule_escalation = "MEDIUM"
    else:              rule_escalation = "LOW"

    rule_escalation_score = escalation_map.get(rule_escalation, 5)
    rule_total = attack_score + severity_score + casualty_score + rule_escalation_score

    # ── LLM score (70% weight) ──
    llm_label_map = {"LOW": 15, "MEDIUM": 40, "HIGH": 65, "CRITICAL": 85}

    if llm_results:
        llm_score = llm_label_map.get(llm_results.get("risk_label", "MEDIUM"), 40)
        llm_escalation = llm_results.get("escalation_label", rule_escalation)
    else:
        llm_score = rule_total
        llm_escalation = rule_escalation

    # ── Merge: 30% rule + 70% LLM ──
    final_score = round((rule_total * 0.30) + (llm_score * 0.70), 2)
    final_score = min(100, max(0, final_score))

    # Final escalation (LLM wins if available)
    final_escalation = llm_escalation if llm_results else rule_escalation

    # Final label
    if final_score >= 75:   final_label = "CRITICAL"
    elif final_score >= 50: final_label = "HIGH"
    elif final_score >= 25: final_label = "MEDIUM"
    else:                   final_label = "LOW"

    print(f"\n📊 Risk Score: {final_score}/100 ({final_label})")
    print(f"   Rule-based: {round(rule_total, 2)} | LLM: {llm_score} | Merged: {final_score}")

    return {
        "risk_score":      final_score,
        "risk_label":      final_label,
        "escalation":      final_escalation,
        "rule_score":      round(rule_total, 2),
        "llm_score":       llm_score,
        "breakdown": {
            "attack_component":     round(attack_score,    2),
            "severity_component":   round(severity_score,  2),
            "casualty_component":   round(casualty_score,  2),
            "escalation_component": round(rule_escalation_score, 2),
        },
        "final_killed":    killed,
        "final_injured":   injured,
        "final_attacks":   attacks,
    }

# ============================================================
# STEP 6 — BUILD FINAL OUTPUT
# ============================================================

def build_output(live_pages, ner_stats, cluster_summaries, risk):
    # Top events feed — from NER articles
    rss_events = []
    try:
        with open(NER_FILE, "r", encoding="utf-8") as f:
            ner_data = json.load(f)

        for item in ner_data:
            if not item.get("conflict"):
                continue

            text  = (item.get("article_text") or item.get("clean_text") or "").lower()
            title = item.get("title", "")

            # Event type
            etype = "general"
            if any(k in text for k in ["airstrike", "air strike", "bombing"]): etype = "airstrike"
            elif any(k in text for k in ["missile", "ballistic", "rocket"]):   etype = "missile_attack"
            elif any(k in text for k in ["drone", "uav"]):                      etype = "drone_attack"
            elif any(k in text for k in ["nuclear", "uranium"]):                etype = "nuclear"
            elif any(k in text for k in ["sanction", "diplomacy", "deal"]):     etype = "diplomatic"
            elif any(k in text for k in ["naval", "ship", "fleet", "strait"]):  etype = "naval"

            killed  = extract_max_number(text, NUMBER_PATTERNS)
            injured = extract_max_number(text, INJURED_PATTERNS)

            # Severity
            sev = 5
            if killed > 50: sev = 9
            elif killed > 10: sev = 8
            elif any(k in text for k in ["nuclear"]): sev = 10
            elif any(k in text for k in ["airstrike", "missile", "killed"]): sev = 8
            elif any(k in text for k in ["wounded", "attack", "explosion"]): sev = 7
            elif any(k in text for k in ["warns", "threatens"]): sev = 5
            elif any(k in text for k in ["talks", "deal", "diplomacy"]): sev = 3

            # Actors from NER
            actors = [a["name"] for a in item.get("top_actors", [])[:3] if a.get("name")]

            # Locations from NER
            locations = [l["name"] for l in item.get("top_locations", [])[:2] if l.get("name")]

            # Geo from NER
            geo = None
            for loc in item.get("top_locations", []):
                if loc.get("geo") and loc["geo"].get("lat"):
                    geo = loc["geo"]
                    break

            rss_events.append({
                "source":     item.get("source", ""),
                "title":      title[:150],
                "summary":    text[:300],
                "url":        item.get("url", ""),
                "published":  item.get("published_at", "") or item.get("published_date", ""),
                "event_type": etype,
                "severity":   sev,
                "killed":     killed,
                "injured":    injured,
                "actors":     actors,
                "locations":  locations,
                "geo":        geo,
                "sentiment":  item.get("actor_sentiment", {}),
            })

    except Exception as e:
        print(f"⚠ Event feed error: {e}")

    # Sort by severity
    rss_events = sorted(rss_events, key=lambda x: -x["severity"])

    aggregate = {
        "total_events":        len(rss_events),
        "total_attacks":       risk["final_attacks"],
        "total_killed":        risk["final_killed"],
        "total_injured":       risk["final_injured"],
        "avg_severity":        ner_stats.get("avg_severity", 0),
        "escalation_level":    risk["escalation"],
        "event_type_counts":   ner_stats.get("event_type_counts", {}),
        "actor_counts":        ner_stats.get("actor_counts", {}),
        "location_counts":     ner_stats.get("location_counts", {}),
        "severity_by_date":    ner_stats.get("severity_by_date", {}),
        "volume_by_date":      ner_stats.get("volume_by_date", {}),
        "casualties_by_date":  ner_stats.get("casualties_by_date", {}),
        "source_counts":       ner_stats.get("source_counts", {}),
        "actor_sentiment":     ner_stats.get("actor_sentiment", {}),
        "last_updated":        datetime.now(timezone.utc).isoformat(),
    }

    return {
        "aggregate":            aggregate,
        "risk":                 risk,
        "wiki_stats":           live_pages,
        "rss_events":           rss_events,
        "cluster_summaries":    cluster_summaries,
        "events_with_casualties": ner_stats.get("events_with_casualties", []),
        "generated_at":         datetime.now(timezone.utc).isoformat(),
    }

# ============================================================
# MAIN
# ============================================================

def run():
    print("=" * 60)
    print("🔴 SAIG LIVE TRACKER STARTING")
    print("=" * 60)

    # Step 1 — Scrape only live tracker pages
    print("\n⬡ STEP 1: Scraping live tracker pages...")
    live_pages = scrape_live_pages()

    # Step 2 — Extract from existing NER file
    print("\n⬡ STEP 2: Extracting from existing NER data...")
    ner_stats = extract_from_ner()

    # Step 3 — Extract cluster summaries
    print("\n⬡ STEP 3: Reading cluster summaries...")
    cluster_summaries = extract_from_clusters()

    # Step 4 — LLM scoring
    print("\n⬡ STEP 4: LLM scoring...")
    llm_results, _ = llm_score_events(live_pages, ner_stats)

    # Step 5 — Risk score (rule + LLM merged)
    print("\n⬡ STEP 5: Computing merged risk score...")
    risk = compute_risk_score(ner_stats, live_pages, llm_results)

    # Step 6 — Build final output
    print("\n⬡ STEP 6: Building final output...")
    output = build_output(live_pages, ner_stats, cluster_summaries, risk)

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*60}")
    print(f"✅ LIVE STATS SAVED → {OUTPUT_FILE}")
    print(f"{'='*60}")
    print(f"  Articles processed : {ner_stats.get('total_articles', 0)}")
    print(f"  Events in feed     : {len(output['rss_events'])}")
    print(f"  Cluster summaries  : {len(cluster_summaries)}")
    print(f"  Live pages scraped : {len(live_pages)}")
    print(f"  Total Killed       : {risk['final_killed']}")
    print(f"  Total Injured      : {risk['final_injured']}")
    print(f"  Avg Severity       : {ner_stats.get('avg_severity', 0)}/10")
    print(f"  Escalation         : {risk['escalation']}")
    print(f"  Risk Score         : {risk['risk_score']}/100 ({risk['risk_label']})")
    print(f"{'='*60}")

if __name__ == "__main__":
    run()