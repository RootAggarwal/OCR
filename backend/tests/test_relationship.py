from ner.extractor import extract_entities
from kg.relationship_extractor import extract_relationships
from kg.neo4j_client import Neo4jClient
from kg.relationship_writer import RelationshipWriter


text = """
Well KJ#25 was drilled in Kanjirangudi.
The well encountered Andimadam Formation at 4200m.
The well subsequently produced HC gas.
"""


# 1. Extract entities
entities = extract_entities(text)

print("Entities:")

for entity in entities:
    print(
        f"{entity['text']} -> "
        f"{entity['label']} "
        f"(score={entity['score']:.3f})"
    )


# 2. For this test, make sure required entities exist
client = Neo4jClient()

with client.driver.session(database=client.database) as session:
    session.run("""
        MERGE (w:Well {name: "KJ#25"})
        MERGE (f:Formation {name: "Andimadam"})
        MERGE (h:Hydrocarbon {name: "HC gas"})
    """)

client.close()


# 3. Extract relationships
relationships = extract_relationships(text, [
    {
        "text": "KJ#25",
        "label": "WELL",
        "score": 1.0,
    },
    {
        "text": "Andimadam",
        "label": "FORMATION",
        "score": 1.0,
    },
    {
        "text": "HC gas",
        "label": "HYDROCARBON",
        "score": 1.0,
    },
])

print("\nRelationships:")

for relationship in relationships:
    print(
        f"{relationship['source']} "
        f"--{relationship['relationship']}--> "
        f"{relationship['target']}"
    )


# 4. Write relationships to Neo4j
writer = RelationshipWriter()

for relationship in relationships:
    writer.create_relationship(
        well_name=relationship["source"],
        relationship=relationship["relationship"],
        target_name=relationship["target"],
        target_label=relationship["target_label"],
        document_name="KJ-25-test",
        page_number=1,
    )

writer.close()

print("\nRelationships written to Neo4j.")