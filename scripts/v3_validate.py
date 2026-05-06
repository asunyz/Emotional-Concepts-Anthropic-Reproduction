"""
Cognitive v3 — Pure validation primitives (no model dependencies).

Used by:
  - generate_trajectories_v3.py (during generation, retry on failure)
  - sanity_check_v3.py (post-hoc check on disk)

Kept stdlib-only so the post-hoc check runs anywhere without GPU deps.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Banned content
# ---------------------------------------------------------------------------

# Cognitive-concept stems. Substring (case-insensitive) match catches
# morphological variants: "uncertainty", "surprises", "confirmation", etc.
# All 9 concept words MUST be banned in every story regardless of which
# trajectory it belongs to — concept words anywhere in the text leak into
# the activation we're trying to extract.
BANNED_STEMS = [
    "curious",    # curiosity, curiously
    "uncertain",  # uncertainty, uncertainly
    "confident",  # confidence, confidently
    "surpris",    # surprise, surprised, surprising, surprisingly, surprises
    "bored",
    "boring",
    "boredom",
    "stubborn",   # stubbornly, stubbornness
    "enlighten",  # enlightened, enlightening, enlightenment
    "confus",     # confused, confusing, confusion
    "confirm",    # confirmed, confirms, confirming, confirmation
]

# Forbidden meta-headers that would leak the assigned stage labels.
HEADER_PATTERNS = [
    re.compile(r"\bstate\s*:", re.I),
    re.compile(r"\btrajectory\s*:", re.I),
    re.compile(r"\bpathway\s*:", re.I),
    re.compile(r"\bcognitive\s+state\s*:", re.I),
    re.compile(r"\bstage\s*:", re.I),
]

# Word-count tolerance bands. The design specifies 40-60 / 40-60 / 80-100;
# we accept ±50% to absorb LLM imprecision about word counts.
WORD_COUNT_TOLERANCES = {
    "P1": (25, 90),
    "P2": (25, 90),
    "P3": (50, 150),
}

# Block extraction. DOTALL so block content can span lines.
P_BLOCK_RE = {
    "P1": re.compile(r"<P1>\s*(.*?)\s*</P1>", re.DOTALL | re.IGNORECASE),
    "P2": re.compile(r"<P2>\s*(.*?)\s*</P2>", re.DOTALL | re.IGNORECASE),
    "P3": re.compile(r"<P3>\s*(.*?)\s*</P3>", re.DOTALL | re.IGNORECASE),
}


def parse_blocks(text: str) -> dict[str, str] | None:
    """Extract <P1>/<P2>/<P3> contents.

    Returns None if any block is missing, duplicated, or out of order.
    """
    blocks = {}
    positions = {}
    for tag, pat in P_BLOCK_RE.items():
        matches = list(pat.finditer(text))
        if len(matches) != 1:
            return None
        blocks[tag] = matches[0].group(1).strip()
        positions[tag] = matches[0].start()
    if not (positions["P1"] < positions["P2"] < positions["P3"]):
        return None
    return blocks


def word_count(s: str) -> int:
    return len(s.split())


def find_banned_words(text: str) -> list[str]:
    lower = text.lower()
    return [stem for stem in BANNED_STEMS if stem in lower]


def find_headers(text: str) -> list[str]:
    return [pat.pattern for pat in HEADER_PATTERNS if pat.search(text)]


def validate_story(text: str, neg: bool = False) -> tuple[bool, list[str], dict | None]:
    """Validate a generated story. Returns (ok, reasons, parsed_blocks).

    NEG is held to the same banned-words rule (cognitive words must not appear).
    """
    reasons = []
    blocks = parse_blocks(text)
    if blocks is None:
        reasons.append("missing_or_misordered_blocks")
        return False, reasons, None

    for tag, (lo, hi) in WORD_COUNT_TOLERANCES.items():
        wc = word_count(blocks[tag])
        if wc < lo or wc > hi:
            reasons.append(f"{tag}_wc={wc}_(allowed_{lo}-{hi})")

    body = " ".join(blocks.values())
    banned = find_banned_words(body)
    if banned:
        reasons.append(f"banned_words={banned}")

    headers = find_headers(body)
    if headers:
        reasons.append(f"headers={headers}")

    return (not reasons), reasons, blocks
