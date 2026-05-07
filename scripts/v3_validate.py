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

# Banned content. Substring (case-insensitive) match catches morphological
# variants. Two layers:
# (1) The 9 concept words themselves — these MUST never appear since they
#     directly leak the assigned label.
# (2) "Feeling-state" words — banning these forces the model to express the
#     cognitive state through action and dialogue, not labeled emotion. The
#     v3 redesign uses this layer to push vectors from surface to functional.
BANNED_STEMS = [
    # --- Layer 1: 9 cognitive concept stems ---
    "curious",     # curiosity, curiously
    "uncertain",   # uncertainty, uncertainly
    "confident",   # confidence, confidently
    "surpris",     # surprise, surprised, surprising, surprisingly, surprises
    "bored",
    "boring",
    "boredom",
    "stubborn",    # stubbornly, stubbornness
    "enlighten",   # enlightened, enlightening, enlightenment
    "confus",      # confused, confusing, confusion
    "confirm",     # confirmed, confirms, confirming, confirmation
    # --- Layer 2: feeling-state words (force show-don't-tell) ---
    "felt",        # "Maria felt..."
    "feeling",     # "with a feeling of..."
    "emotion",     # "the emotion of..."
    "wondered",    # as inner-state verb; questions are still allowed via dialogue
    "pondered",
    "mused",
    "intrigued",
    "fascinated",
    "baffled",
    "perplexed",
    "dumbfounded",
    "shocked",
    "stunned",
    "startled",
    "astounded",
    "thrilled",
    "doubted",     # action-of-doubting as verb
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

# Markdown-style stage headers. Earlier <P1>/</P1> tag pairs led to systematic
# generation failures: Qwen consistently confused "</P3>" as the closing tag
# for P2 (apparently associating "P3" with the closing role rather than the
# third paragraph). Markdown headers have no open/close pair to mismatch.
STAGE_HEADER_RE = re.compile(
    r"^[ \t]*#{1,3}[ \t]*(prior|discovery|reaction)[ \t]*$",
    re.MULTILINE | re.IGNORECASE,
)
STAGE_TO_TAG = {"prior": "P1", "discovery": "P2", "reaction": "P3"}


def parse_blocks(text: str) -> dict[str, str] | None:
    """Extract Prior / Discovery / Reaction sections demarcated by markdown
    headers (## Prior / ## Discovery / ## Reaction).

    Returns dict keyed by P1/P2/P3 (for compatibility with downstream code),
    or None if the three headers are missing, duplicated, or out of order.
    """
    matches = list(STAGE_HEADER_RE.finditer(text))
    if len(matches) != 3:
        return None
    found = [m.group(1).lower() for m in matches]
    if found != ["prior", "discovery", "reaction"]:
        return None

    blocks: dict[str, str] = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[m.end():end].strip()
        blocks[STAGE_TO_TAG[m.group(1).lower()]] = content
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
