import json
from neo4j import GraphDatabase

# =========================
# CONFIG
# =========================

INPUT_FILE = "step6_output.json"

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "865739@pj"

# =========================
# CONNECT
# =========================

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

# =========================
# CREATE CONSTRAINTS (RUN ONCE)
# =========================

def create_constraints():
    with driver.session(database="neo4j") as session:
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Actor) REQUIRE a.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (l:Location) REQUIRE l.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Article) REQUIRE a.url IS UNIQUE")

# =========================
# INSERT DATA
# =========================

def insert_article(tx, article):

    url = article.get("url", "")
    title = article.get("title", "")

    tx.run("""
        MERGE (a:Article {url: $url})
        SET a.title = $title
    """, url=url, title=title)

def insert_actor(tx, actor_name):
    tx.run("""
        MERGE (a:Actor {name: $name})
    """, name=actor_name)

def insert_location(tx, location_name, geo):
    tx.run("""
        MERGE (l:Location {name: $name})
        SET l.lat = $lat, l.lon = $lon
    """, name=location_name,
       lat=geo.get("lat") if geo else None,
       lon=geo.get("lon") if geo else None)


def link_actor_event(tx, actor, event_id):
    tx.run("""
        MATCH (a:Actor {name: $actor})
        MATCH (e:Event {id: $event_id})
        MERGE (a)-[:INVOLVED_IN]->(e)
    """, actor=actor, event_id=event_id)

def link_event_location(tx, event_id, location):
    tx.run("""
        MATCH (e:Event {id: $event_id})
        MATCH (l:Location {name: $location})
        MERGE (e)-[:OCCURRED_IN]->(l)
    """, event_id=event_id, location=location)

def link_actor_target(tx, actor, target):
    tx.run("""
        MATCH (a1:Actor {name: $actor})
        MATCH (a2:Actor {name: $target})
        MERGE (a1)-[:TARGETS]->(a2)
    """, actor=actor, target=target)

def link_article_actor(tx, url, actor):
    tx.run("""
        MATCH (art:Article {url: $url})
        MATCH (a:Actor {name: $actor})
        MERGE (a)-[:MENTIONED_IN]->(art)
    """, url=url, actor=actor)
    
    
def insert_event(tx, event_id, event, article):
    tx.run("""
        MERGE (e:Event {id: $id})
        SET e.actor = $actor,
            e.action = $action,
            e.target = $target,
            e.type = $type,
            e.date = $date,
            e.time = $time
    """,
    id=event_id,
    actor=event.get("actor"),
    action=event.get("action"),
    target=event.get("target"),
    type=event.get("type"),
    date=article.get("published_date"),
    time=article.get("published_time")
    )

# =========================
# MAIN PROCESS
# =========================

def process():

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    create_constraints()

    with driver.session() as session:

        for idx, article in enumerate(data):

            print(f"[GRAPH] Processing article {idx}")

            url = article.get("url", "")
            actors_data = article.get("top_actors", [])
            locations_data = article.get("top_locations", [])
            actor_names = [a["name"] for a in actors_data]
            location_names = [l["name"] for l in locations_data]
            events = article.get("relations", {}).get("llm_validated", [])

# 🔥 AUTO-GENERATE EVENTS IF EMPTY
            if not events:
                for i in range(len(actor_names)):
                    for j in range(i + 1, len(actor_names)):
                        events.append({
                            "actor": actor_names[i],
                            "action": "associated_with",
                            "target": actor_names[j],
                            "location": location_names[0] if location_names else None,
                            "type": "inferred"
            })
           
    # create pairwise interactions
                for i in range(len(actor_names)):
                    for j in range(i + 1, len(actor_names)):
                         events.append({
                            "actor": actor_names[i],
                            "action": "associated_with",
                            "target": actor_names[j],
                            "location": location_names[0] if location_names else None,
                            "type": "inferred"
            })
            
            # 1. ARTICLE
            session.execute_write(insert_article, article)

            # 2. ACTORS
            for actor in actors_data:
                name = actor["name"]
                
                session.execute_write(insert_actor, name)
                session.execute_write(link_article_actor, url, name)

            # 3. LOCATIONS
            for loc in locations_data:
                name = loc["name"]
                geo = loc.get("geo")
                session.execute_write(insert_location, name, geo)

            # 4. EVENTS
            for i, event in enumerate(events):

                event_id = f"{idx}_{i}"

                
                session.execute_write(insert_event, event_id, event, article)

                actor = event.get("actor")
                target = event.get("target")
                location = event.get("location")

                if actor:
                    session.execute_write(insert_actor, actor)
                    session.execute_write(link_actor_event, actor, event_id)

                if target:
                    session.execute_write(insert_actor, target)
                    session.execute_write(link_actor_target, actor, target)

                if location:
                    session.execute_write(insert_location, location, {})
                    session.execute_write(link_event_location, event_id, location)

    print("\n✅ GRAPH SUCCESSFULLY CREATED IN NEO4J")

# =========================
# RUN
# =========================

if __name__ == "__main__":
    process()