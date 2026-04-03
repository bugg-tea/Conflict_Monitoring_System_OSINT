from neo4j import GraphDatabase

# =========================
# CONFIG
# =========================

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "865739@pj"

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

# =========================
# STEP 1: ADD TEMPORAL DATA
# =========================

def add_temporal_properties(tx):
    tx.run("""
    MATCH (e:Event)<-[:INVOLVED_IN]-(a:Actor)
    MATCH (art:Article)<-[:MENTIONED_IN]-(a)

    WHERE art.published_date IS NOT NULL

    SET e.date = art.published_date,
        e.time = art.published_time
    """)

# =========================
# STEP 2: CREATE TIME LINKS
# =========================

def create_temporal_links(tx):
    tx.run("""
    MATCH (a:Actor)-[:INVOLVED_IN]->(e1:Event),
          (a)-[:INVOLVED_IN]->(e2:Event)
    WHERE e1.date < e2.date

    MERGE (e1)-[:PRECEDES]->(e2)
    """)

# =========================
# STEP 3: CREATE CO-OCCURRENCE
# =========================

def create_actor_graph(tx):

    # Create CO_OCCURS_WITH relationships
    tx.run("""
    MATCH (e:Event)<-[:INVOLVED_IN]-(a1:Actor)
    MATCH (e)<-[:INVOLVED_IN]-(a2:Actor)
    WHERE a1.name < a2.name

    MERGE (a1)-[:CO_OCCURS_WITH]->(a2)
    """)

# =========================
# STEP 4: PROJECT GRAPH (NO APOC)
# =========================

def project_graph(tx):

    # Drop if exists
    tx.run("""
    CALL gds.graph.exists('actorGraph')
    YIELD exists
    WHERE exists
    CALL gds.graph.drop('actorGraph')
    YIELD graphName
    RETURN graphName
    """)

    # Create projection
    tx.run("""
    CALL gds.graph.project(
        'actorGraph',
        'Actor',
        {
            CO_OCCURS_WITH: {
                orientation: 'UNDIRECTED'
            }
        }
    )
    """)

# =========================
# STEP 5: COMMUNITY DETECTION
# =========================

def run_louvain(tx):
    tx.run("""
    CALL gds.louvain.write('actorGraph', {
        writeProperty: 'community'
    })
    """)

# =========================
# STEP 6: CREATE COMMUNITY LABELS
# =========================

def create_community_labels(tx):

    tx.run("""
    MATCH (a:Actor)
    WHERE a.community IS NOT NULL

    SET a.group = "bloc_" + toString(a.community)
    """)

# =========================
# STEP 7: ACTIVITY SCORE
# =========================

def compute_temporal_activity(tx):

    tx.run("""
    MATCH (a:Actor)-[:INVOLVED_IN]->(e:Event)
    WHERE e.date IS NOT NULL

    WITH a, count(e) AS activity
    SET a.activity_score = activity
    """)

# =========================
# MAIN PIPELINE
# =========================

def process():

    with driver.session() as session:

        print("🔄 Adding temporal properties...")
        session.execute_write(add_temporal_properties)

        print("🔄 Creating time links...")
        session.execute_write(create_temporal_links)

        print("🔄 Building actor interaction graph...")
        session.execute_write(create_actor_graph)

        # 🔥 IMPORTANT CHECK
        count = session.run("""
        MATCH ()-[r:CO_OCCURS_WITH]->()
        RETURN count(r) AS count
        """).single()["count"]

        print(f"✅ CO_OCCURS_WITH relationships: {count}")

        if count == 0:
            raise Exception("❌ No CO_OCCURS_WITH relationships found. Fix your data first.")

        print("🔄 Projecting graph...")
        session.execute_write(project_graph)

        print("🔄 Running community detection...")
        session.execute_write(run_louvain)

        print("🔄 Creating community labels...")
        session.execute_write(create_community_labels)

        print("🔄 Computing activity scores...")
        session.execute_write(compute_temporal_activity)

    print("\n✅ TEMPORAL + COMMUNITY GRAPH READY")

# =========================
# RUN
# =========================

if __name__ == "__main__":
    process()