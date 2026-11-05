from kg.relationship_extractor import extract_relationships


text = """
Well KJ#25 was drilled in Kanjirangudi.
The well encountered Andimadam Formation at 4200m.
The well subsequently produced HC gas.
"""


entities = [
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
]


relationships = extract_relationships(text, entities)

print("Relationships:")

for relationship in relationships:
    print(
        f"{relationship['source']} "
        f"--{relationship['relationship']}--> "
        f"{relationship['target']}"
    )