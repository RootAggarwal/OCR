from .neo4j_client import Neo4jClient


ENTITY_LABELS = {
    "WELL": "Well",
    "FIELD": "Field",
    "FORMATION": "Formation",
    "RESERVOIR": "Reservoir",
    "FACILITY": "Facility",
    "LOCATION": "Location",
    "HYDROCARBON": "Hydrocarbon",
    "LITHOLOGY": "Lithology",
    "DEPTH": "Depth",
    "DATE": "Date",
}


class EntityWriter:
    def __init__(self):
        self.client = Neo4jClient()

    def write_entities(self, entities, document_name, page_number):
        with self.client.driver.session(
            database=self.client.database
        ) as session:

            for entity in entities:
                label = ENTITY_LABELS.get(entity["label"])

                if not label:
                    continue

                session.run(
                    f"""
                    MERGE (e:{label} {{name: $name}})
                    SET e.document = $document,
                        e.page = $page,
                        e.score = $score
                    """,
                    name=entity["text"],
                    document=document_name,
                    page=page_number,
                    score=entity["score"],
                )

    def close(self):
        self.client.close()