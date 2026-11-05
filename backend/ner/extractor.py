import re

from models.model import model
from .labels import ENTITY_LABELS
from .rules import apply_domain_rules


def extract_entities(text: str):
    chunk_size = 2000
    all_entities = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]

        entities = model.predict_entities(
            chunk,
            ENTITY_LABELS,
            threshold=0.3,
        )

        all_entities.extend(entities)

    # Add obvious well identifiers
    for match in re.finditer(
        r"\b[A-Z]{1,5}\s*#\s*\d+\b|\bKJAH\b",
        text,
        re.IGNORECASE,
    ):
        all_entities.append({
            "text": match.group(),
            "label": "WELL",
            "score": 1.0,
        })

    # Add obvious geological formations
    for match in re.finditer(
        r"\b([A-Z][A-Za-z]+)\s+Formation\b",
        text,
    ):
        all_entities.append({
            "text": match.group(1),
            "label": "FORMATION",
            "score": 1.0,
        })

    return apply_domain_rules(all_entities)