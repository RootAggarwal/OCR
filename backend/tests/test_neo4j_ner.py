from ner.extractor import extract_entities
from knowledge_graph.entity_writer import EntityWriter


text = """
Well KJ#25 was drilled in Kanjirangudi.
The well encountered Andimadam Formation at 4200m.
The well subsequently produced HC gas.
"""

entities = extract_entities(text)

for entity in entities:
    print(
        f"{entity['text']} -> "
        f"{entity['label']} "
        f"(score={entity['score']:.3f})"
    )


writer = EntityWriter()

writer.write_entities(
    entities,
    document_name="KJ-25-test",
    page_number=1,
)

writer.close()

print("\nEntities written to Neo4j.")