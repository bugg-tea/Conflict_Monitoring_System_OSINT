import json
import re
from collections import defaultdict

import spacy
from geopy.geocoders import Nominatim
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer, util

# =========================
# CONFIG
# =========================

INPUT_FILE  = "step5_output.json"
OUTPUT_FILE = "step6_output.json"

# Minimum text length to attempt full NER + event extraction.
# Articles below this use title-only NER and skip heavy processing.
MIN_TEXT_LENGTH = 80

# =========================
# LOAD MODELS  (once, at import time)
# =========================

print("[INIT] Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

print("[INIT] Loading geocoder...")
geolocator = Nominatim(user_agent="geo_pipeline_v2")

print("[INIT] Loading sentiment model...")
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1,          # CPU
    truncation=True,
    max_length=512,
)

print("[INIT] Loading cross-encoder...")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("[INIT] Loading sentence transformer...")
st_model = SentenceTransformer("all-MiniLM-L6-v2")

print("[INIT] All models loaded.\n")

# =========================
# SERIALIZATION HELPER
# =========================

def convert_to_serializable(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    if hasattr(obj, "item"):          # numpy scalar
        return obj.item()
    return obj

# =========================
# ENTITY NORMALIZATION MAP
# =========================

ENTITY_MAP = {
    "US":              "United States",
    "USA":             "United States",
    "U.S.":            "United States",
    "U.S":             "United States",
    "UK":              "United Kingdom",
    "U.K.":            "United Kingdom",
    "UAE":             "United Arab Emirates",
    "Iran":            "Iran",
    "Iranian":         "Iran",
    "IRGC":            "IRGC",
    "Pentagon":        "Pentagon",
    "Houthi":          "Houthis",
    "Houthis":         "Houthis",
    "IDF":             "Israel Defense Forces",
    "Hamas":           "Hamas",
    "Hezbollah":       "Hezbollah",
}

def normalize_entity(ent_text):
    """Normalize common abbreviations and variants."""
    cleaned = ent_text.strip()
    return ENTITY_MAP.get(cleaned, cleaned)

# =========================
# GEO CODING  (with cache)
# =========================

_geo_cache = {}

def get_coordinates(place):
    if place in _geo_cache:
        return _geo_cache[place]
    try:
        loc = geolocator.geocode(place, timeout=5)
        if loc:
            result = {"lat": loc.latitude, "lon": loc.longitude}
            _geo_cache[place] = result
            return result
    except Exception:
        pass
    _geo_cache[place] = None
    return None

# =========================
# ENTITY EXTRACTION
# =========================

def extract_entities(text):
    """
    Run spaCy NER on text and return categorised entity lists.
    Deduplication is applied per category.
    """
    doc = nlp(text[:5000])   # cap to avoid OOM on very long texts

    seen   = {"people": set(), "organizations": set(), "locations": set()}
    result = {"people": [],    "organizations": [],    "locations": []}

    for ent in doc.ents:
        val = normalize_entity(ent.text)

        if ent.label_ == "PERSON" and val not in seen["people"]:
            seen["people"].add(val)
            result["people"].append(val)

        elif ent.label_ == "ORG" and val not in seen["organizations"]:
            seen["organizations"].add(val)
            result["organizations"].append(val)

        elif ent.label_ in ("GPE", "LOC") and val not in seen["locations"]:
            seen["locations"].add(val)
            result["locations"].append(val)

    return result

# =========================
# ENTITY SCORING  (TF-IDF + heuristics)
# =========================

def compute_entity_scores(text, title, entity_list):
    """
    Score each entity by a weighted combination of:
      - Frequency in text
      - Presence in title
      - Position of first occurrence (earlier → higher score)
      - TF-IDF weight
    """
    if not entity_list or not text:
        return {}

    scores    = defaultdict(float)
    words     = text.split()
    total_w   = max(len(words), 1)

    # Build TF-IDF over the single document
    try:
        vectorizer    = TfidfVectorizer(stop_words="english")
        tfidf         = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores  = dict(zip(feature_names, tfidf.toarray()[0]))
    except Exception:
        tfidf_scores = {}

    for ent in entity_list:
        freq            = len(re.findall(
            r'\b' + re.escape(ent) + r'\b', text, re.IGNORECASE
        ))
        freq_score      = freq / total_w
        title_score     = 1.0 if ent.lower() in title.lower() else 0.0
        first_occ       = text.lower().find(ent.lower())
        position_score  = 1 - (first_occ / max(len(text), 1)) \
                          if first_occ != -1 else 0.0
        tfidf_score     = tfidf_scores.get(ent.lower(), 0.0)

        scores[ent] = (
            0.4 * freq_score
            + 0.3 * title_score
            + 0.2 * position_score
            + 0.1 * tfidf_score
        )

    return scores

# =========================
# CROSS-ENCODER RERANKING
# =========================

def rerank_entities(text, entities, top_k=5):
    """
    Use cross-encoder to score entity relevance against article text.
    Returns list of (entity, score) tuples sorted descending.
    Falls back to returning all entities with score 0 if list is tiny.
    """
    if not entities:
        return []
    if len(entities) == 1:
        return [(entities[0], 0.0)]

    # Cross-encoder needs at least 2 chars per entity
    valid = [e for e in entities if len(e.strip()) >= 2]
    if not valid:
        return []

    try:
        pairs  = [[text[:500], ent] for ent in valid]
        scores = cross_encoder.predict(pairs)
        ranked = sorted(zip(valid, scores),
                        key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
    except Exception as e:
        print(f"  [RERANK ERROR] {e}")
        return [(e, 0.0) for e in valid[:top_k]]

# =========================
# SENTIMENT
# =========================

def get_actor_sentiment(text, actors):
    """
    Run sentiment analysis with actor name prepended to short text excerpt.
    Handles short texts gracefully.
    """
    result  = {}
    excerpt = text[:250].strip()

    for actor, _ in actors:
        try:
            input_text = f"{actor}: {excerpt}"
            res        = sentiment_model(input_text[:512])[0]
            result[actor] = res["label"]
        except Exception:
            result[actor] = "UNKNOWN"

    return result

# =========================
# RULE-BASED RELATION EXTRACTION  (improved patterns)
# =========================

# Named actors to anchor patterns — avoids matching random single words
_ACTOR_PATTERN = (
    r'(?:Iran(?:ian)?|US|United States|Israel(?:i)?|Pentagon|IRGC|'
    r'Houthi(?:s)?|Biden|Trump|Netanyahu|Hamas|Hezbollah|'
    r'CENTCOM|White House|Congress|Kremlin|Russia(?:n)?|China|'
    r'Saudi Arabia|NATO)'
)

RELATION_PATTERNS = [
    # Attack events
    (
        rf'({_ACTOR_PATTERN}[\w\s]{{0,15}})'
        r'\s+(?:has\s+)?(?:attacked|struck|bombed|shelled|hit)\s+'
        rf'(the\s+)?({_ACTOR_PATTERN}[\w\s]{{0,20}})',
        "attack"
    ),
    # Strike / launch events
    (
        rf'({_ACTOR_PATTERN}[\w\s]{{0,15}})'
        r'\s+(?:has\s+)?(?:launched|fired|deployed)\s+'
        r'(?:a\s+)?(?:missile|drone|strike|airstrike|rocket|bomb|attack)s?'
        r'\s+(?:at|against|on|toward)\s+'
        rf'(the\s+)?({_ACTOR_PATTERN}[\w\s]{{0,20}})',
        "strike"
    ),
    # Retaliation
    (
        rf'({_ACTOR_PATTERN}[\w\s]{{0,15}})'
        r'\s+(?:has\s+)?retaliated\s+(?:against|on)\s+'
        rf'({_ACTOR_PATTERN}[\w\s]{{0,20}})',
        "retaliation"
    ),
    # Warning / threat
    (
        rf'({_ACTOR_PATTERN}[\w\s]{{0,15}})'
        r'\s+(?:has\s+)?(?:warned|threatened|vowed\s+to\s+(?:strike|attack))\s+'
        rf'({_ACTOR_PATTERN}[\w\s]{{0,20}})',
        "threat"
    ),
    # Sanctions
    (
        rf'({_ACTOR_PATTERN}[\w\s]{{0,15}})'
        r'\s+(?:has\s+)?sanctioned\s+'
        rf'({_ACTOR_PATTERN}[\w\s]{{0,20}})',
        "sanction"
    ),
    # Negotiations / talks
    (
        rf'({_ACTOR_PATTERN}[\w\s]{{0,15}})'
        r'\s+(?:has\s+)?(?:agreed\s+to\s+)?(?:negotiate|hold\s+talks|'
        r'reached\s+a\s+deal|signed\s+(?:a\s+)?(?:deal|agreement))\s+'
        r'with\s+'
        rf'({_ACTOR_PATTERN}[\w\s]{{0,20}})',
        "negotiation"
    ),
]

def extract_relations_rule(text):
    """
    Extract structured relations using domain-specific regex patterns.
    Returns a deduplicated list of {subject, object, type} dicts.
    """
    relations = []
    seen      = set()

    for pattern, rel_type in RELATION_PATTERNS:
        try:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Patterns may have 2 or 3 groups depending on optional article
                if len(match) == 3:
                    subj = match[0].strip()
                    obj  = match[2].strip()
                elif len(match) == 2:
                    subj = match[0].strip()
                    obj  = match[1].strip()
                else:
                    continue

                # Clean up trailing punctuation / whitespace
                subj = re.sub(r'[,;:\s]+$', '', subj)[:60]
                obj  = re.sub(r'[,;:\s]+$', '', obj)[:60]

                key = (subj.lower(), obj.lower(), rel_type)
                if key not in seen and subj.lower() != obj.lower() \
                        and len(subj) >= 2 and len(obj) >= 2:
                    seen.add(key)
                    relations.append({
                        "subject": subj,
                        "object":  obj,
                        "type":    rel_type,
                    })
        except re.error:
            continue

    return relations

# =========================
# SPACY EVENT EXTRACTION  (replaces flan-t5 LLM)
# =========================

# Verbs that indicate a conflict or geopolitical event
CONFLICT_VERBS = {
    "attack", "strike", "bomb", "fire", "launch", "retaliate",
    "deploy", "invade", "shell", "threaten", "intercept", "target",
    "kill", "arrest", "sanction", "warn", "condemn", "seize",
    "detain", "shoot", "destroy", "hit", "capture", "withdraw",
    "negotiate", "sign", "agree", "reject", "impose", "blockade",
    "escalate", "assassinate", "execute",
}

def extract_events_spacy(text, actors, locations):
    """
    Use spaCy dependency parsing to extract structured events.
    Each event has: actor, action, target, location, type.

    This replaces the flan-t5 LLM call which was unreliable and
    didn't produce valid JSON consistently.
    """
    if not text:
        return {"events": []}

    doc    = nlp(text[:4000])
    events = []
    seen   = set()

    # Build lookup sets for faster membership testing
    actor_names = set()
    for a in actors:
        name = a[0] if isinstance(a, tuple) else str(a)
        actor_names.add(name.lower())

    loc_names = set()
    for l in locations:
        name = l[0] if isinstance(l, tuple) else str(l)
        loc_names.add(name.lower())

    for sent in doc.sents:
        for token in sent:
            # Only process conflict-relevant verbs
            if token.lemma_.lower() not in CONFLICT_VERBS:
                continue

            # ── Find grammatical subject ─────────────────
            subj_token = next(
                (c for c in token.children
                 if c.dep_ in ("nsubj", "nsubjpass")),
                None
            )
            # Expand to full noun phrase if possible
            if subj_token:
                subj_np  = subj_token.text
                # Try to get the full noun chunk
                for chunk in sent.noun_chunks:
                    if subj_token in chunk:
                        subj_np = chunk.text
                        break
            else:
                # Try compound subject via conj
                subj_np = None

            if not subj_np:
                continue

            # ── Find grammatical object ──────────────────
            obj_token = next(
                (c for c in token.children
                 if c.dep_ in ("dobj", "pobj", "attr", "nsubjpass")),
                None
            )
            obj_np = ""
            if obj_token:
                obj_np = obj_token.text
                for chunk in sent.noun_chunks:
                    if obj_token in chunk:
                        obj_np = chunk.text
                        break

            # Also look inside prepositional phrases
            # e.g. "fired missiles AT Iran"
            if not obj_np:
                for child in token.children:
                    if child.dep_ == "prep":
                        prep_obj = next(
                            (c for c in child.children
                             if c.dep_ == "pobj"),
                            None
                        )
                        if prep_obj:
                            obj_np = prep_obj.text
                            break

            # ── Find location in sentence ────────────────
            loc_text = ""
            for ent in sent.ents:
                if ent.label_ in ("GPE", "LOC"):
                    loc_text = ent.text
                    break

            # ── Deduplication ────────────────────────────
            key = (
                subj_np.lower()[:30],
                token.lemma_.lower(),
                obj_np.lower()[:30],
            )
            if key in seen:
                continue
            seen.add(key)

            events.append({
                "actor":    subj_np.strip(),
                "action":   token.text.strip(),
                "target":   obj_np.strip(),
                "location": loc_text.strip(),
                "type":     "conflict_event",
            })

            if len(events) >= 15:   # cap per article
                break

        if len(events) >= 15:
            break

    return {"events": events}


def validate_events_rule(events, text):
    """
    Validate extracted events by checking that the action verb
    or actor name appears in the source text.
    Replaces the LLM-based validation which was unreliable.
    """
    validated = []
    text_lower = text.lower()

    for e in events:
        actor  = (e.get("actor")  or "").lower()
        action = (e.get("action") or "").lower()

        # Keep the event if at least actor OR action appears in text
        if actor in text_lower or action in text_lower:
            validated.append(e)

    return validated

# =========================
# CONFLICT DETECTION
# =========================

CONFLICT_KEYWORDS = {
    "war", "attack", "missile", "conflict", "strike", "bomb",
    "explosion", "retaliation", "airstrike", "escalation",
    "invasion", "military operation", "nuclear", "drone", "troops",
    "combat", "soldier", "casualt", "killed", "wounded",
}

def detect_conflict(text):
    """Return True if text contains conflict-related keywords."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in CONFLICT_KEYWORDS)

# =========================
# NORMALIZE EVENTS OUTPUT
# =========================

def normalize_events_output(data):
    if not isinstance(data, dict):
        return {"events": []}
    events = data.get("events", [])
    if not isinstance(events, list):
        return {"events": []}

    clean = []
    for e in events:
        if isinstance(e, dict):
            clean.append({
                "actor":    str(e.get("actor",    "")),
                "action":   str(e.get("action",   "")),
                "target":   str(e.get("target",   "")),
                "location": str(e.get("location", "")),
                "type":     str(e.get("type",     "conflict_event")),
            })
    return {"events": clean}

# =========================
# MAIN PIPELINE
# =========================

def process():

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    results       = []
    total         = len(data)
    full_ner      = 0
    title_only    = 0
    skipped_empty = 0

    print(f"\n{'='*55}")
    print(f"  FILE 5 — NER + Relation + Event Extraction Pipeline")
    print(f"  Total articles : {total}")
    print(f"{'='*55}\n")

    for idx, item in enumerate(data, 1):

        title = item.get("title", "").strip()
        text  = item.get("article_text", "").strip()

        # Combine title + text so short articles still get something
        combined = (title + ". " + text).strip() if title else text

        print(f"\n[{idx}/{total}] {title[:65]}")
        print(f"  text_len={len(text)}  combined_len={len(combined)}")

        # ── Case 1: Nothing to work with ─────────────────
        if len(combined) < 10:
            skipped_empty += 1
            print(f"  ⚠️  Skipping — no usable text")

            results.append({
                **item,
                "entities":       {"people": [], "organizations": [], "locations": []},
                "top_actors":     [],
                "top_locations":  [],
                "actor_sentiment": {},
                "relations": {
                    "rule_based":    [],
                    "llm_raw":       [],
                    "llm_validated": [],
                },
                "conflict": detect_conflict(title),
            })
            continue

        # ── NER — always run on combined (title + text) ──
        entities      = extract_entities(combined)
        all_entities  = (
            entities["people"]
            + entities["organizations"]
            + entities["locations"]
        )

        # ── Entity scoring ────────────────────────────────
        scores = compute_entity_scores(combined, title, all_entities)

        # ── Cross-encoder reranking ───────────────────────
        actor_candidates    = list(set(
            entities["people"] + entities["organizations"]
        ))
        location_candidates = list(set(entities["locations"]))

        actors    = rerank_entities(combined, actor_candidates,    top_k=5)
        locations = rerank_entities(combined, location_candidates, top_k=5)

        # ── Geo coding ────────────────────────────────────
        geo_locations = []
        for loc, score in locations:
            geo_locations.append({
                "name":  loc,
                "score": float(score),
                "geo":   get_coordinates(loc),
            })

        # ── Case 2: Short text — skip heavy event extraction
        if len(text) < MIN_TEXT_LENGTH:
            title_only += 1
            print(f"  ℹ️  Title-only NER (text too short for events)")

            results.append({
                **item,
                "entities":       entities,
                "top_actors":     [{"name": a, "score": float(s)}
                                   for a, s in actors],
                "top_locations":  geo_locations,
                "actor_sentiment": {},   # skip sentiment for short articles
                "relations": {
                    "rule_based":    extract_relations_rule(combined),
                    "llm_raw":       [],
                    "llm_validated": [],
                },
                "conflict": detect_conflict(combined),
            })
            continue

        # ── Case 3: Full processing ───────────────────────
        full_ner += 1

        # Sentiment
        sentiment = get_actor_sentiment(text, actors)

        # Rule-based relations
        rule_relations = extract_relations_rule(text)

        # spaCy event extraction (replaces flan-t5)
        events_output  = extract_events_spacy(text, actors, locations)
        events_output  = normalize_events_output(events_output)
        raw_events     = events_output.get("events", [])

        # Rule-based validation (replaces LLM validation)
        validated_events = validate_events_rule(raw_events, text)

        # Conflict flag
        conflict_flag = detect_conflict(text)

        print(f"  ✅ actors={len(actors)}  locs={len(locations)}"
              f"  rule_rel={len(rule_relations)}"
              f"  events={len(raw_events)}"
              f"  validated={len(validated_events)}"
              f"  conflict={conflict_flag}")

        enriched = item.copy()
        enriched.update({
            "entities":       entities,
            "top_actors":     [{"name": a, "score": float(s)}
                               for a, s in actors],
            "top_locations":  geo_locations,
            "actor_sentiment": sentiment,
            "relations": {
                "rule_based":    rule_relations,
                "llm_raw":       raw_events,       # spaCy events, same field name
                "llm_validated": validated_events,  # rule-validated events
            },
            "conflict": conflict_flag,
        })

        results.append(enriched)

    # ── Save ──────────────────────────────────────────────
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(convert_to_serializable(results), f,
                  indent=2, ensure_ascii=False)

    # ── Stats ─────────────────────────────────────────────
    articles_with_relations = sum(
        1 for r in results
        if (r.get("relations", {}).get("rule_based")
            or r.get("relations", {}).get("llm_raw"))
    )
    articles_with_actors = sum(
        1 for r in results if r.get("top_actors")
    )
    conflict_count = sum(
        1 for r in results if r.get("conflict")
    )

    print(f"\n{'='*55}")
    print(f"  PIPELINE STATS")
    print(f"{'='*55}")
    print(f"  Total articles         : {total}")
    print(f"  ✅ Full NER + events    : {full_ner}")
    print(f"  ℹ️  Title-only NER       : {title_only}")
    print(f"  ⚠️  Skipped (empty)      : {skipped_empty}")
    print(f"  🔗 With relations/events: {articles_with_relations}")
    print(f"  👤 With actors found    : {articles_with_actors}")
    print(f"  ⚔️  Conflict flagged     : {conflict_count}")
    print(f"  📦 Output saved         : {OUTPUT_FILE}")
    print(f"{'='*55}\n")

# =========================
# RUN
# =========================

if __name__ == "__main__":
    process()