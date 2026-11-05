from ner.extractor import extract_entities
from kg.relationship_extractor import extract_relationships


text = """
The location KJAH was drilled vertically as KJ#25 to a depth of 4230m
and terminated within the Andimadam Formation.

The K/T boundary, Nannilam Formation, Kudavasal Shale,
Bhuvanagiri Formation, Sattapadi Shale and Andimadam Formation
tops were encountered.

The Nannilam Formation was encountered in the interval 1918-1989m
and consists of sandstone and shale.

The Bhuvanagiri Formation was encountered in the interval
2081-2836m and consists of sandstone with shale.

The Andimadam Formation was encountered in the interval
3576-4230m and consists of sandstone and shale alternations.

During drilling, hydrocarbon shows were observed.

The object-V was reviewed and tested conventionally.
After 1st LN2 application, the well flowed HC gas.

The status of the well is gas producer from BS-10 sand
in Bhuvanagiri Formation.
"""


# ---------------------------------------------------------
# 1. NER
# ---------------------------------------------------------

entities = extract_entities(text)

print("\nENTITIES")
print("-" * 50)

for entity in entities:
    print(
        f"{entity['text']} -> "
        f"{entity['label']} "
        f"(score={entity['score']:.3f})"
    )


# ---------------------------------------------------------
# 2. Relationship extraction
# ---------------------------------------------------------

relationships = extract_relationships(text, entities)

print("\nRELATIONSHIPS")
print("-" * 50)

for relationship in relationships:
    print(
        f"{relationship['source']} "
        f"--{relationship['relationship']}--> "
        f"{relationship['target']}"
    )