from .neo4j_client import Neo4jClient


class RelationshipWriter:
    def __init__(self):
        self.client = Neo4jClient()

    def create_relationship(
        self,
        well_name,
        relationship,
        target_name,
        target_label,
        document_name,
        page_number,
    ):
        with self.client.driver.session(
            database=self.client.database
        ) as session:

            session.run(
                f"""
                MATCH (w:Well {{name: $well}})
                MATCH (t:{target_label} {{name: $target}})
                MERGE (w)-[r:{relationship}]->(t)
                SET r.document = $document,
                    r.page = $page
                """,
                well=well_name,
                target=target_name,
                document=document_name,
                page=page_number,
            )

    def close(self):
        self.client.close()