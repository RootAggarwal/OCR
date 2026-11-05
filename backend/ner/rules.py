import re


# Known terms from the ONGC WCR domain
LITHOLOGY_TERMS = {
    "sandstone",
    "sandstones",
    "shale",
    "limestone",
    "siltstone",
    "mudstone",
    "dolomite",
    "claystone",
    "conglomerate",
}


HYDROCARBON_TERMS = {
    "oil",
    "gas",
    "hydrocarbon",
    "hydrocarbons",
    "hc gas",
    "oil and gas",
}


def is_depth(text: str) -> bool:
    """Return True only for values that look like geological depths."""

    lower = text.lower().strip()

    # Definitely not depth
    invalid_terms = [
        "psi",
        "md/cp",
        "mdd",
        "ohm",
        "ωm",
        "latitude",
        "longitude",
        "wgs",
        "hole",
        "casing",
        "bean",
        "depth point",
        "depth points",
    ]

    if any(term in lower for term in invalid_terms):
        return False

    # Must contain metres
    if not re.search(r"\bm\b", lower):
        return False

    # Examples:
    # 4200m
    # 4221m MSL
    # 2083-2836m
    # 3580.5-3581.1m
    return bool(
        re.fullmatch(
            r"\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?\s*m(?:\s*msl)?",
            lower,
        )
    )


def is_well(text: str) -> bool:
    """Return True for common well identifiers."""

    text = text.strip().upper()

    # KJ#25, KJ # 25, KJ#3 etc.
    if re.fullmatch(r"[A-Z]{1,5}\s*#\s*\d+", text):
        return True

    # KJAH is the well identifier used in this report
    if text == "KJAH":
        return True

    # B-CY-KJ-16 is a location/well reference in the report,
    # but keep it as a candidate well identifier.
    if re.fullmatch(r"[A-Z]-[A-Z]{2}-[A-Z]{2}-\d+", text):
        return True

    return False


def clean_entity(entity):
    text = entity["text"].strip()
    label = entity["label"].upper()

    if len(text) < 2:
        return None

    lower = text.lower()

    # ---------------------------------------------------------
    # WELL
    # ---------------------------------------------------------
    if label == "WELL":
        # Reject generic words
        if lower in {
            "well",
            "nearby wells",
            "object-v",
        }:
            return None

        # Rig / logging-tool identifiers
        if any(term in lower for term in [
            "e-1400",
            "hdil",
            "zdl",
            "mrex",
            "rtex",
            "xmac",
            "cal-",
            "-gr",
            "-sp",
            "-cn",
        ]):
            return None

        # Core/sample identifiers are not wells
        if re.fullmatch(r"cc\s*#\s*\d+", lower):
            return None

        # Other known non-well abbreviations
        if text.upper() in {
            "PCL",
            "IL-2459",
            "KJ ML",
        }:
            return None

        # Keep only plausible well identifiers
        if not is_well(text):
            return None

    # ---------------------------------------------------------
    # DEPTH
    # ---------------------------------------------------------
    elif label == "DEPTH":
        if not is_depth(text):
            return None

    # ---------------------------------------------------------
    # HYDROCARBON
    # ---------------------------------------------------------
    elif label == "HYDROCARBON":
        if lower not in HYDROCARBON_TERMS:
            return None

    # ---------------------------------------------------------
    # LITHOLOGY
    # ---------------------------------------------------------
    elif label == "LITHOLOGY":
        if lower not in LITHOLOGY_TERMS:

            # Geological formation names such as
            # Portonovo Shale should still be retained.
            if not any(term in lower for term in [
                "shale",
                "sandstone",
                "limestone",
                "siltstone",
                "mudstone",
                "dolomite",
            ]):
                return None

    # ---------------------------------------------------------
    # FORMATION
    # ---------------------------------------------------------
    elif label == "FORMATION":
        if lower in {
            "formation",
            "vertical",
            "annulus",
            "conventional",
            "ii",
            "iii",
            "iv",
            "interval",
            "msl",
            "wgs-84",
            "wgs- 84",
        }:
            return None

        if any(term in lower for term in [
            "section",
            "depth points",
            "depth point",
            "receiver",
            "shot",
            "m. secs",
            "rtEx".lower(),
        ]):
            return None

        # CC#1 / CC#2 are core identifiers, not formations
        if re.fullmatch(r"cc\s*#\s*\d+", lower):
            return None

    # ---------------------------------------------------------
    # FIELD
    # ---------------------------------------------------------
    elif label == "FIELD":
        # GLiNER frequently interprets document headings
        # as FIELD entities.
        if any(term in lower for term in [
            "section",
            "data",
            "results",
            "process",
            "review",
            "details",
            "activities",
            "personnel",
            "analysis",
            "records",
            "prognosis",
            "compliance",
            "consumption",
            "complications",
            "electrolog",
            "geological",
        ]):
            return None

        if lower in {
            "location",
            "object",
            "interval",
            "formation",
        }:
            return None

    # ---------------------------------------------------------
    # LOCATION
    # ---------------------------------------------------------
    elif label == "LOCATION":
        # These are clearly not locations
        if lower in {
            "kjah",
            "hole",
            "formation",
            "well",
        }:
            return None

        # Andimadam is a formation name in this WCR
        if lower == "andimadam":
            entity["label"] = "FORMATION"

        # Well identifiers should not become locations
        if is_well(text):
            return None

        # Oil & gas field names
        if lower.endswith(" field"):
            entity["label"] = "FIELD"

    # ---------------------------------------------------------
    # RESERVOIR
    # ---------------------------------------------------------
    elif label == "RESERVOIR":
        # Generic "well" is not a reservoir
        if lower == "well":
            return None

    # ---------------------------------------------------------
    # FACILITY
    # ---------------------------------------------------------
    elif label == "FACILITY":
        # ONGC is an organisation/company, not a facility.
        if lower == "ongc":
            return None

    return entity


def apply_domain_rules(entities):
    cleaned = []

    for entity in entities:
        entity = clean_entity(entity)

        if entity is not None:
            cleaned.append(entity)

    return cleaned