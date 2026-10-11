"""
Follow-up query helpers — let a short, referential query ("Where exactly?",
"How much area?", "Was this visible in SAR?") lean on the PREVIOUS turn's
answer instead of forcing the user to retype the full context every time.

Phase 15 built hop 1 (-> grounding). Phase 16 (this file, extended) adds
the intent classifier that distinguishes all three hops from
uisuggestions.docx's "Ask Follow-up" chain:

    "Has the built-up area increased?"   -> change_vqa            (turn 1)
    "Where exactly?"                     -> locate    -> grounding (hop 1)
    "How much area?"                     -> area      -> area calc (hop 2)
    "Was this visible in SAR?"           -> sar_check -> grounding on SAR (hop 3)

Phase 15 also shipped hop 1 with a real bug: agent/router.py's
classify_query() only recognises "where is" (not bare "where exactly?" or
"show it") as a grounding-intent keyword, so a literal "Where exactly?"
follow-up would default to VQA and never reach the grounding handler at
all — the phrase-inference code existed but was frequently unreachable.
followup_intent() below is now checked BEFORE classify_query() in
api/query.py's run_query() precisely to fix that: when a follow-up intent
is detected, the task is set directly, bypassing keyword classification.

HONESTY CONTRACT:
This is NOT conversational memory, NOT a dialogue-state model, and NOT an
LLM call. It is rule-based pattern matching over plain text the frontend
already has (the previous turn's query + answer, and — for hops 2/3 —
the previous turn's grounding box/image, which the frontend now also
sends back) — nothing is persisted server-side beyond a single request.
  1. followup_intent() — classifies which of the three hops (if any) a
     query looks like, purely from surface patterns.
  2. is_followup_query() — the original Phase 15 catch-all heuristic,
     kept for backward compatibility with any caller still using it
     directly; followup_intent() is the richer, preferred entry point.
  3. infer_target_phrase() — pulls a plausible short noun phrase out of
     the PREVIOUS answer text (checking the PS's own representative
     vocabulary first, then falling back to a generic heuristic) so a
     subsequent grounding call has something concrete to search for.
All of these will sometimes guess wrong on unusual phrasing — that is an
accepted trade-off for a rule-based prototype, consistent with
agent/router.py and agent/nlp_extract.py's own "do NOT over-engineer the
agent" approach.
"""
from __future__ import annotations

import re
from typing import Literal, Optional

FollowupIntent = Literal["locate", "area", "sar_check"]

# Checked in this order — most specific first — by followup_intent().
_SAR_CHECK_PATTERNS = [
    r"^(and )?was (it|this|that) (visible|seen)( in| on)? (the )?(sar|radar)\??$",
    r"^(and )?(is (it|this) visible) (in|on) (the )?(sar|radar)\??$",
    r"^(and )?what about( the)? (sar|radar)\??$",
    r"^(and )?(does|did) (sar|radar) (show|confirm|see) (it|this|that)\??$",
    r"^(and )?(check|cross.?check) (this|it) (in|against|with) (the )?(sar|radar)\??$",
]

_AREA_PATTERNS = [
    r"^how much( area| space)?\??$",
    r"^how big( is it| is that)?\??$",
    r"^what('?s| is) the area\??$",
    r"^(and )?what('?s| is) the (approx(imate)?\.? )?(affected )?area\??$",
    r"^(and )?how many (km2|km\^?2|sq ?km|square k(ilo)?meters?)\??$",
]

_FOLLOWUP_PATTERNS = [
    r"^where( exactly)?\??$",
    r"^where is (it|that|this)\??$",
    r"^how much( area)?\??$",
    r"^how big( is it)?\??$",
    r"^show (it|that|me)\??$",
    r"^highlight (it|that)\??$",
    r"^(and )?was (it|this|that) (visible|seen)",
    r"^(and )?(what about|is (it|this) visible) (in|on) (the )?sar",
]

# Very short queries that lean on pronouns/deixis are treated as follow-ups
# even if they don't match a fixed pattern above.
_MAX_STANDALONE_WORDS = 4
_DEICTIC_WORDS = ("it", "that", "this", "there", "exactly")


def is_followup_query(query: str) -> bool:
    q = query.strip().lower()
    if not q:
        return False
    if any(re.match(p, q) for p in _FOLLOWUP_PATTERNS):
        return True
    words = q.rstrip("?.!").split()
    return len(words) <= _MAX_STANDALONE_WORDS and any(w in _DEICTIC_WORDS for w in words)


def followup_intent(query: str) -> Optional[FollowupIntent]:
    """Classify which follow-up hop `query` looks like, or None if it
    doesn't read as a follow-up at all. Order matters: SAR-check and area
    patterns are checked first because they are strictly more specific
    than the generic "locate" catch-all (e.g. "how much area?" would
    otherwise also satisfy is_followup_query's short-deictic-query check).

    Callers should only trust this when a previous turn actually exists
    (i.e. `previous_answer` was supplied) — a bare "where exactly?" with
    no prior context is not a follow-up, it's just an underspecified
    question, and should fail normal validation instead of being routed
    here.
    """
    q = query.strip().lower()
    if not q:
        return None
    if any(re.match(p, q) for p in _SAR_CHECK_PATTERNS):
        return "sar_check"
    if any(re.match(p, q) for p in _AREA_PATTERNS):
        return "area"
    if is_followup_query(q):
        return "locate"
    return None


# PS-representative vocabulary checked first — matches the kind of nouns
# the problem statement's own sample queries use ("built-up area", "water
# body", etc.), so a real hero-demo answer resolves cleanly.
_KNOWN_TARGETS = [
    "built-up area", "built up area", "water body", "flooded area",
    "flooded surface", "flooding", "vegetation", "bare soil",
    "agricultural land", "urban area", "water", "cloud",
]

_STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "are", "in", "on", "at", "of",
    "this", "that", "these", "those", "and", "or", "to", "region", "area",
    "image", "scene", "between", "detected", "visible", "between", "date",
    "dates", "confidence", "high", "medium", "low", "note",
}


def infer_target_phrase(previous_answer: str) -> str | None:
    """Best-effort grounding phrase pulled from the previous turn's free-text
    answer. Returns None if nothing usable was found — callers should fall
    back to the literal (current) query text in that case, never guess
    silently past that point."""
    if not previous_answer:
        return None
    a = previous_answer.lower()

    for target in _KNOWN_TARGETS:
        if target in a:
            return target

    words = [w.strip(".,;:()") for w in a.split()]
    words = [w for w in words if w and w not in _STOPWORDS]
    return " ".join(words[:3]) if words else None
