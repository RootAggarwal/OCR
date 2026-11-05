import re


def extract_relationships(text, entities):
    relationships = []

    wells = [
        e for e in entities
        if e["label"] == "WELL"
    ]

    formations = [
        e for e in entities
        if e["label"] == "FORMATION"
    ]

    hydrocarbons = [
        e for e in entities
        if e["label"] == "HYDROCARBON"
    ]

    # Use the well name that appears explicitly as KJ#25.
    well_name = next(
        (e["text"] for e in wells
         if re.fullmatch(r"KJ\s*#\s*25", e["text"], re.IGNORECASE)),
        None,
    )

    if not well_name and wells:
        well_name = wells[0]["text"]

    if not well_name:
        return relationships

    # ---------------------------------------------------------
    # ENCOUNTERED -> FORMATION
    # ---------------------------------------------------------

    for formation in formations:
        pattern = rf"\b{re.escape(formation['text'])}\b[^.\n]{{0,150}}\bencountered\b"

        if re.search(pattern, text, re.IGNORECASE):
            relationships.append({
                "source": well_name,
                "relationship": "ENCOUNTERED",
                "target": formation["text"],
                "target_label": "Formation",
            })

        pattern = rf"\bencountered\b[^.\n]{{0,150}}\b{re.escape(formation['text'])}\b"

        if re.search(pattern, text, re.IGNORECASE):
            relationships.append({
                "source": well_name,
                "relationship": "ENCOUNTERED",
                "target": formation["text"],
                "target_label": "Formation",
            })

    # ---------------------------------------------------------
    # PRODUCES -> HYDROCARBON
    # ---------------------------------------------------------

    for hydrocarbon in hydrocarbons:
        pattern = rf"\b(produced|producer|producing|flowed)\b[^.\n]{{0,150}}\b{re.escape(hydrocarbon['text'])}\b"

        if re.search(pattern, text, re.IGNORECASE):
            relationships.append({
                "source": well_name,
                "relationship": "PRODUCES",
                "target": hydrocarbon["text"],
                "target_label": "Hydrocarbon",
            })

    # ---------------------------------------------------------
    # Remove duplicates
    # ---------------------------------------------------------

    unique = {}

    for relationship in relationships:
        key = (
            relationship["source"],
            relationship["relationship"],
            relationship["target"],
        )

        unique[key] = relationship

    return list(unique.values())