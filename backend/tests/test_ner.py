from ner.extractor import extract_entities


text = """
Well MH-12 was drilled in MH Field and encountered
Panna Formation at a depth of 3245 m. The well was
subsequently put on production and produced oil and gas.
"""


entities = extract_entities(text)


for entity in entities:
    print(
        f"{entity['text']} -> "
        f"{entity['label']} "
        f"(score={entity['score']:.3f})"
    )