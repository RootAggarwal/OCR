import os

from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv()


class Neo4jClient:

    def __init__(self):
        self.uri = os.getenv("NEO4J_URI")
        self.username = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")

        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password),
        )

        self.driver.verify_connectivity()

    def close(self):
        self.driver.close()

  