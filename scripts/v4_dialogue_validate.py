"""
Cognitive v4 — Dialogue validator (no model dependencies).

Validates a single dialogue text emitted by the model:
  - Strictly alternating Person 1 / Person 2 turns starting with Person 1
  - 6 to 10 total turns (3-5 per speaker)
  - Each turn 5-50 words (1-3 sentences)
  - No banned stems anywhere (case-insensitive substring match for morphological variants)
  - No meta-commentary lines (e.g. "Dialogue:", "[end]", "Person 3:")

Returns (ok, reasons, turns) where:
  ok: True iff all checks pass
  reasons: list of failure reasons (empty if ok)
  turns: list[{"speaker": "P1"|"P2", "text": str}] (best-effort even if invalid)

Stdlib-only.
"""
from __future__ import annotations

import re
from typing import List, Tuple, Dict


TURN_LINE = re.compile(r"^\s*Person\s*(\d+)\s*:\s*(.*)$", re.IGNORECASE)
META_PATTERNS = [
    re.compile(r"^\s*\[?end\]?\s*$", re.I),
    re.compile(r"^\s*dialogue\s*\d*\s*:?\s*$", re.I),
    re.compile(r"^\s*---+\s*$"),
    re.compile(r"^\s*###+", re.I),
]

MIN_TURNS = 6
MAX_TURNS = 10
MIN_WORDS_PER_TURN = 5
MAX_WORDS_PER_TURN = 60


def parse_turns(text: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """Best-effort parse. Returns (turns, parse_warnings)."""
    turns: List[Dict[str, str]] = []
    warnings: List[str] = []
    lines = text.splitlines()
    current_speaker: str | None = None
    current_text: List[str] = []

    def flush():
        if current_speaker is not None:
            turns.append({
                "speaker": current_speaker,
                "text": "\n".join(current_text).strip(),
            })

    for raw in lines:
        line = raw.rstrip()
        # detect meta lines that should not appear
        if any(p.match(line) for p in META_PATTERNS) and line.strip():
            warnings.append(f"meta-line: {line.strip()!r}")
        m = TURN_LINE.match(line)
        if m:
            speaker_num = m.group(1)
            if speaker_num not in ("1", "2"):
                warnings.append(f"unexpected speaker: Person {speaker_num}")
            # close the previous turn before starting a new one
            flush()
            current_speaker = f"P{speaker_num}"
            current_text = [m.group(2)]
        else:
            if current_speaker is None:
                # text before the first Person N: line — usually preamble. Treat as warning.
                if line.strip():
                    warnings.append(f"pre-turn text: {line.strip()[:50]!r}")
            else:
                current_text.append(line)
    flush()
    # strip empty trailing turn
    while turns and not turns[-1]["text"].strip():
        turns.pop()
    return turns, warnings


def check_banned_stems(text: str, banned_stems: List[str]) -> List[str]:
    """Return list of banned stems found (substring case-insensitive)."""
    lower = text.lower()
    hits = []
    for stem in banned_stems:
        if stem.lower() in lower:
            hits.append(stem)
    return hits


def validate_dialogue(
    text: str,
    p1_state: str,
    p2_state: str,
    banned_stems: List[str],
) -> Tuple[bool, List[str], List[Dict[str, str]]]:
    """Validate a single dialogue completion. Returns (ok, reasons, turns)."""
    reasons: List[str] = []
    turns, warnings = parse_turns(text)
    reasons.extend(warnings)

    # Structural checks
    if not turns:
        reasons.append("no Person N: turns found")
        return False, reasons, turns

    if turns[0]["speaker"] != "P1":
        reasons.append(f"first turn is {turns[0]['speaker']}, expected P1")

    # Strict alternation
    for i, t in enumerate(turns):
        expected = "P1" if i % 2 == 0 else "P2"
        if t["speaker"] != expected:
            reasons.append(f"turn {i+1} speaker is {t['speaker']}, expected {expected} (alternation broken)")
            break  # one report is enough; cascading errors not useful

    # Turn count
    if len(turns) < MIN_TURNS:
        reasons.append(f"too few turns: {len(turns)} < {MIN_TURNS}")
    elif len(turns) > MAX_TURNS:
        reasons.append(f"too many turns: {len(turns)} > {MAX_TURNS}")

    # Per-turn word count
    for i, t in enumerate(turns):
        n = len(t["text"].split())
        if n < MIN_WORDS_PER_TURN:
            reasons.append(f"turn {i+1} ({t['speaker']}) too short: {n} words < {MIN_WORDS_PER_TURN}")
        elif n > MAX_WORDS_PER_TURN:
            reasons.append(f"turn {i+1} ({t['speaker']}) too long: {n} words > {MAX_WORDS_PER_TURN}")

    # Banned content (states + custom stems)
    full = "\n".join(t["text"] for t in turns)
    state_words = [p1_state, p2_state]
    state_hits = check_banned_stems(full, state_words)
    if state_hits:
        reasons.append(f"named states leaked: {state_hits}")

    stem_hits = check_banned_stems(full, banned_stems)
    if stem_hits:
        reasons.append(f"banned stems: {stem_hits}")

    ok = not reasons
    return ok, reasons, turns


if __name__ == "__main__":
    # Self-test: P1 confused, P2 thoughtful
    sample = """Person 1: I'm not totally clear on what we're looking at here. Could you walk me through it?

Person 2: Sure. Take your time. Let me set context first — what part is unclear?

Person 1: Wait, the third row of which table? I might have lost track.

Person 2: No rush. Let me restate. The lab data — column three, third row down.

Person 1: Ah, I see it now. So that flag on the value, what does it mean?

Person 2: Good question. It marks values outside the reference range. Does that help?
"""
    ok, reasons, turns = validate_dialogue(
        sample, p1_state="confused", p2_state="thoughtful",
        banned_stems=["confus", "thoughtful"],
    )
    print(f"ok={ok}")
    print(f"reasons={reasons}")
    print(f"turns={len(turns)}")
    for t in turns:
        print(f"  [{t['speaker']}] {t['text'][:60]}...")
