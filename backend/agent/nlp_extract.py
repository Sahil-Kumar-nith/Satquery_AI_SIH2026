"""
Location + time extraction for the query router.

Phase 13 (P0/P1 — closes a gap flagged during review: the router
(agent/router.py) classified TASK type but never surfaced the location
or time information embedded in the query text, even though the query
itself (and the map's drawn point/rectangle/polygon) are what actually
carry location, and phrases like "between June and August" carry time).

Deliberately rule-based/regex, NOT an LLM call — consistent with
router.py's existing "deterministic rules... do NOT over-engineer the
agent" approach, and keeps this dependency-free (no spaCy/NER model to
download mid-hackathon).

SCOPE — be honest about what this is:
- Time extraction: regex over explicit dates, years, month names, and a
  small set of common relative-time phrases ("last year", "this
  monsoon", "between X and Y"). This is pattern matching, not a
  temporal-reasoning model — it will miss unusual phrasing.
- Location extraction: a small gazetteer of India/ISRO-relevant place
  names (matches the hero-demo geography) PLUS a generic
  capitalized-phrase heuristic as a fallback for names not in the list.
  This is NOT a trained NER model and will both miss valid places and
  occasionally flag a capitalized non-place word — surfaced confidence
  is "gazetteer" (higher trust) vs "heuristic" (lower trust) so the UI
  and judges can see which kind of match this was.
- The map's click/rectangle/polygon selection remains the authoritative
  source of location for actually running a model — this module is a
  transparency/UX feature (showing what the query *said* about
  location/time) not a replacement for that geometry.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Small gazetteer biased toward the PS's own geography (ISRO/Cartosat-2S/
# RISAT evaluation set, Assam monsoon hero demo) plus commonly-used
# generic regions. Not exhaustive — extend as needed, don't pretend it's
# a full geocoder.
_GAZETTEER = [
    "assam", "kerala", "mumbai", "delhi", "chennai", "kolkata", "bengaluru",
    "bangalore", "hyderabad", "pune", "ahmedabad", "gujarat", "punjab",
    "rajasthan", "odisha", "bihar", "uttar pradesh", "west bengal",
    "tamil nadu", "andhra pradesh", "telangana", "maharashtra", "goa",
    "himachal pradesh", "uttarakhand", "jammu and kashmir", "ladakh",
    "sundarbans", "brahmaputra", "ganga", "ganges", "western ghats",
    "northeast india", "north east india", "india", "bangladesh",
]

_MONTHS = (
    "january|february|march|april|may|june|july|august|september|"
    "october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)

# Ordered so more specific patterns are tried first.
_TIME_PATTERNS = [
    re.compile(rf"\bbetween\s+({_MONTHS})\s*(?:\d{{4}})?\s+and\s+({_MONTHS})\s*(?:\d{{4}})?\b", re.I),
    re.compile(rf"\b({_MONTHS})\s+(\d{{4}})\b", re.I),
    re.compile(r"\b(19|20)\d{2}\b"),
    re.compile(r"\blast\s+(year|month|monsoon|season|week)\b", re.I),
    re.compile(r"\bthis\s+(year|month|monsoon|season)\b", re.I),
    re.compile(r"\b(before|after|pre[- ]|post[- ])(monsoon|flood|construction)\b", re.I),
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),  # dd/mm/yyyy style dates
]

# Generic capitalized multi-word / single-word phrase, used only as a
# fallback when nothing in the gazetteer matched. Deliberately excludes
# common sentence-initial and query-scaffolding words so "Where is..."
# doesn't get flagged as a place called "Where".
_MONTH_WORDS = {
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec",
}

_STOPWORDS = {
    "where", "what", "which", "how", "has", "did", "is", "are", "the",
    "this", "these", "those", "image", "images", "region", "area",
    "compare", "describe", "highlight", "locate", "assess", "identify",
    "use", "combine", "show", "find", "sar", "optical", "vqa",
} | _MONTH_WORDS


@dataclass
class ExtractionResult:
    locations: list[str] = field(default_factory=list)
    location_source: str = "none"   # "gazetteer" | "heuristic" | "gazetteer+heuristic" | "none"
    time_expressions: list[str] = field(default_factory=list)
    note: str = ""


def extract_locations(query: str) -> tuple[list[str], str]:
    q_lower = query.lower()
    gazetteer_hits = sorted({name for name in _GAZETTEER if re.search(rf"\b{re.escape(name)}\b", q_lower)})

    heuristic_hits: list[str] = []
    for match in re.finditer(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b", query):
        phrase = match.group(1)
        if phrase.lower() in _STOPWORDS:
            continue
        if match.start() == 0:
            # Skip a capitalized first word — too likely to just be
            # normal sentence capitalization, not a place name.
            continue
        if phrase.lower() in gazetteer_hits:
            continue
        heuristic_hits.append(phrase)

    locations = gazetteer_hits + [h for h in heuristic_hits if h.lower() not in gazetteer_hits]
    if gazetteer_hits and heuristic_hits:
        source = "gazetteer+heuristic"
    elif gazetteer_hits:
        source = "gazetteer"
    elif heuristic_hits:
        source = "heuristic"
    else:
        source = "none"
    return locations, source


def extract_time_expressions(query: str) -> list[str]:
    hits: list[str] = []
    for pattern in _TIME_PATTERNS:
        for match in pattern.finditer(query):
            span = match.group(0).strip()
            if span and span not in hits:
                hits.append(span)
    return hits


def extract_location_time(query: str) -> ExtractionResult:
    locations, source = extract_locations(query)
    time_expressions = extract_time_expressions(query)

    if not locations and not time_expressions:
        note = (
            "No location or time expression detected in the query text. This is "
            "pattern-matching over the query string only — it does not mean the "
            "query lacks spatial/temporal scope (the uploaded image / map "
            "selection may already define that)."
        )
    else:
        parts = []
        if locations:
            parts.append(f"location candidate(s) via {source} match")
        if time_expressions:
            parts.append("time expression(s) via pattern match")
        note = (
            "Detected " + " and ".join(parts) + ". Rule-based extraction, not a "
            "trained NER/temporal model — verify against the actual uploaded "
            "imagery/map selection before treating this as ground truth scope."
        )

    return ExtractionResult(
        locations=locations,
        location_source=source,
        time_expressions=time_expressions,
        note=note,
    )
