"""
Parse stained_<concept>.html files to extract top-N highest-activation tokens
per concept (with their projection values).

Each token in HTML is wrapped like:
    <span style="background:rgba(...);" title="curious: +1.575">Dr</span>

We pull the token text + projection value, sort by value, output top-N.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


TOKEN_RE = re.compile(
    r'title="(?P<concept>[^:]+):\s*(?P<value>[+-][0-9.]+)"[^>]*>(?P<text>[^<]*)</span>'
)


def parse_staining(path: Path) -> tuple[str, list[tuple[float, str]]]:
    """Return (concept_name, [(value, token_text)]) sorted by value desc."""
    html = path.read_text()
    tokens: list[tuple[float, str]] = []
    concept = None
    for m in TOKEN_RE.finditer(html):
        c = m.group("concept")
        if concept is None:
            concept = c
        v = float(m.group("value"))
        t = m.group("text").replace("&#x27;", "'").replace("&amp;", "&")
        tokens.append((v, t))
    return concept or "?", tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staining-dir", required=True,
                    help="dir with stained_<concept>.html files")
    ap.add_argument("--top-n", type=int, default=15)
    ap.add_argument("--output", default=None,
                    help="output JSON path; default = <staining-dir>/top_tokens.json")
    args = ap.parse_args()

    sd = Path(args.staining_dir)
    htmls = sorted(sd.glob("stained_*.html"))
    print(f"found {len(htmls)} HTML files")

    summary: dict[str, dict] = {}
    print("\n=== Top-{} highest-activation tokens per concept ===\n".format(args.top_n))
    for h in htmls:
        concept, tokens = parse_staining(h)
        if not tokens:
            continue
        # also compute basic stats
        vals = np.array([v for v, _ in tokens])
        n_pos_strong = int((vals > 1.0).sum())
        n_neg = int((vals < 0.0).sum())
        n_total = len(vals)
        # top-N (highest) and bottom-N (lowest)
        sorted_desc = sorted(tokens, reverse=True)
        sorted_asc = sorted(tokens)
        top = sorted_desc[:args.top_n]
        bot = sorted_asc[:args.top_n]

        summary[concept] = {
            "n_tokens": n_total,
            "max": float(vals.max()),
            "min": float(vals.min()),
            "mean": float(vals.mean()),
            "n_strong_positive": n_pos_strong,
            "n_negative": n_neg,
            "top_15": [{"tok": t.strip().replace("\n", "\\n"), "value": round(v, 3)}
                       for v, t in top],
            "bottom_15": [{"tok": t.strip().replace("\n", "\\n"), "value": round(v, 3)}
                          for v, t in bot],
        }

        print(f"--- {concept} ---")
        print(f"  total tokens: {n_total},  range: [{vals.min():+.2f}, {vals.max():+.2f}],  mean: {vals.mean():+.2f}")
        print(f"  strong positive (>1.0): {n_pos_strong},  negative: {n_neg}")
        print(f"  TOP-{args.top_n} (most concept-aligned tokens):")
        for v, t in top:
            tt = t.strip()
            if not tt:
                tt = repr(t)
            print(f"    {v:+.3f}  {tt!r}")
        print()

    out_path = Path(args.output) if args.output else sd / "top_tokens.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
