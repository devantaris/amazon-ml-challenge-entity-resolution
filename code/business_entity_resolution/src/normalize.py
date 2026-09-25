"""
Text Normalization for Business Entity Resolution.

Handles:
- Unicode NFKD normalization & unidecode (for transliteration flattening).
- Lowercasing, punctuation stripping, whitespace collapsing.
- Bidirectional legal suffix normalization (Corp/Corporation, Pvt/Private, Ltd/Limited, etc.).
- Address abbreviations (Rd/Road, St/Street, Ave/Avenue, etc.).
- Extraction of PIN / Postal codes, street numbers, and landmark references.
- Token sets and acronyms for rapid similarity indexing and feature calculation.
"""

import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple


# Precompiled regular expressions for speed
RE_WHITESPACE = re.compile(r"\s+")
RE_AMPERSAND = re.compile(r"&")
RE_PUNCTUATION = re.compile(r"[^\w\s]")
RE_POSTAL_CODE = re.compile(r"\b(\d{5}(?:-\d{4})?|\b[1-9]\d{5})\b")
RE_STREET_NUMBER = re.compile(r"^\s*(\d+[a-zA-Z]?(?:-\d+[a-zA-Z]?)?)\b")
RE_LANDMARK = re.compile(
    r"\b(?:near|opp|opposite|behind|adjacent to|beside|in front of|next to)\s+([^,]+)",
    re.IGNORECASE,
)

# Common legal and corporate abbreviations canonical mapping
LEGAL_MAPPINGS: Dict[str, str] = {
    "corporation": "corp",
    "corporate": "corp",
    "incorporated": "inc",
    "incorporation": "inc",
    "limited": "ltd",
    "private": "pvt",
    "company": "co",
    "enterprises": "ent",
    "enterprise": "ent",
    "technologies": "tech",
    "technology": "tech",
    "services": "svc",
    "service": "svc",
    "solutions": "soln",
    "solution": "soln",
    "international": "intl",
    "industries": "ind",
    "industry": "ind",
    "manufacturing": "mfg",
    "management": "mgmt",
    "associates": "assoc",
    "association": "assoc",
    "group": "grp",
    "department": "dept",
    "foundation": "fdn",
    "holdings": "hldgs",
    "holding": "hldg",
    "development": "dev",
    "llc": "llc",
    "llp": "llp",
    "pllc": "pllc",
    "gmbh": "gmbh",
    "sa": "sa",
    "sarl": "sarl",
    "sas": "sas",
    "bv": "bv",
    "nv": "nv",
    "pty": "pty",
}

# Common address abbreviations canonical mapping
ADDRESS_MAPPINGS: Dict[str, str] = {
    "street": "st",
    "str": "st",
    "road": "rd",
    "avenue": "ave",
    "boulevard": "blvd",
    "bvd": "blvd",
    "drive": "dr",
    "lane": "ln",
    "court": "ct",
    "circle": "cir",
    "highway": "hwy",
    "expressway": "expy",
    "parkway": "pkwy",
    "floor": "fl",
    "flr": "fl",
    "apartment": "apt",
    "suite": "ste",
    "building": "bldg",
    "block": "blk",
    "sector": "sec",
    "district": "dist",
    "post office": "po",
    "p o": "po",
    "opposite": "opp",
    "near": "nr",
    "adjacent": "adj",
    "station": "stn",
    "centre": "ctr",
    "center": "ctr",
    "plaza": "plz",
    "square": "sq",
}

# Stopwords for candidate indexing (very common terms that produce bloated posting lists)
COMMON_NAME_STOPWORDS: Set[str] = {
    "and", "the", "of", "in", "for", "on", "at", "to", "by", "a", "an",
    "co", "corp", "inc", "ltd", "pvt", "llc", "grp", "svc", "soln"
}

# Domain extensions regex
RE_DOMAIN = re.compile(r"\.(?:com|org|net|in|co|io|biz|info|gov|edu)(?:\.in)?\b", re.IGNORECASE)

# US state abbreviations mapping to full names and vice-versa
US_STATES: Dict[str, str] = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi", "mo": "missouri",
    "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new hampshire", "nj": "new jersey",
    "nm": "new mexico", "ny": "new york", "nc": "north carolina", "nd": "north dakota", "oh": "ohio",
    "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont",
    "va": "virginia", "wa": "washington", "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming"
}


def basic_clean(text: str) -> str:
    """Normalize unicode, lowercase, standardize ampersands and collapse whitespace."""
    if not text or not isinstance(text, str):
        return ""
    # NFKD unicode normalization
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = RE_AMPERSAND.sub(" and ", text)
    return text


def normalize_business_name(name: str) -> Tuple[str, List[str], str, str]:
    """
    Normalizes a business name.

    Returns
    -------
    Tuple[str, List[str], str, str]
        (cleaned_name, token_list, initials_acronym, compressed_name)
    """
    cleaned = basic_clean(name)
    cleaned = RE_DOMAIN.sub(" ", cleaned)
    # Remove punctuation
    cleaned = RE_PUNCTUATION.sub(" ", cleaned)
    tokens = [t for t in RE_WHITESPACE.split(cleaned) if t]

    # Map legal abbreviations
    mapped_tokens = [LEGAL_MAPPINGS.get(t, t) for t in tokens]
    name_str = " ".join(mapped_tokens)
    acronym = "".join(t[0] for t in mapped_tokens if t)
    compressed = "".join(mapped_tokens)

    return name_str, mapped_tokens, acronym, compressed


def normalize_business_address(address: str) -> Tuple[str, List[str], Optional[str], Optional[str], Optional[str]]:
    """
    Normalizes a business address.

    Returns
    -------
    Tuple[str, List[str], Optional[str], Optional[str], Optional[str]]
        (cleaned_address, token_list, postal_code, street_number, landmark)
    """
    cleaned = basic_clean(address)

    # Extract landmark before punctuation stripping
    landmark_match = RE_LANDMARK.search(cleaned)
    landmark = landmark_match.group(1).strip() if landmark_match else None

    # Extract postal code
    postal_match = RE_POSTAL_CODE.search(cleaned)
    postal_code = postal_match.group(1).replace("-", "") if postal_match else None

    # Extract leading street number
    street_num_match = RE_STREET_NUMBER.search(cleaned)
    street_number = street_num_match.group(1).strip() if street_num_match else None

    # Remove punctuation
    cleaned = RE_PUNCTUATION.sub(" ", cleaned)
    tokens = [t for t in RE_WHITESPACE.split(cleaned) if t]

    # Map address abbreviations
    mapped_tokens = [ADDRESS_MAPPINGS.get(t, t) for t in tokens]
    address_str = " ".join(mapped_tokens)

    return address_str, mapped_tokens, postal_code, street_number, landmark


def normalize_country(country: str) -> str:
    """Standardizes country label safely without hard-coded exclusivity."""
    if not country or not isinstance(country, str):
        return ""
    c = country.strip().upper()
    if c in ("UNITED STATES", "USA", "U.S.A.", "U.S.", "US"):
        return "US"
    if c in ("INDIA", "IND", "IN"):
        return "INDIA"
    if c in ("FRANCE", "FR", "FRA"):
        return "FRANCE"
    return c
