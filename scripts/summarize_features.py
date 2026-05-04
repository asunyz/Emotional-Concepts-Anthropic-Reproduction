"""Phase 4-finale: join contrastive ranks + epistemic probe + conflate test.

Reads:
    candidates_layer_<L>_surprise_vs__neutral.csv  (Phase 3 main contrast)
    candidates_layer_<L>_afraid_vs__neutral.csv    (Phase 3 conflate control)
    epistemic_layer_<L>.csv                        (Phase 4 paired test)

Writes one final table with the verdict for each candidate:
    final_layer_<L>.csv

Verdict rules:
    epistemic         — subcat_coverage >= 4 and paired_t > 2.5
    surprise_affective — specificity > 0.70 and not epistemic
                          (i.e. it's surprise-specific vs afraid, but
                           doesn't generalize to non-emotional violations)
    conflate          — specificity in [0.40, 0.60] (lights up similarly
                          on afraid stories — same conflation as mean-diff)
    other             — everything else

specificity = mean_surprise / (mean_surprise + mean_afraid)
              with both clipped to >= 0 and a tiny epsilon to avoid div-by-zero.

Usage:
    python scripts/summarize_features.py --layer 30
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EPS = 1e-6


def load_csv_indexed(path: Path, key: str = "feature_id") -> dict[int, dict]:
    out: dict[int, dict] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[int(row[key])] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", required=True, type=int)
    ap.add_argument("--out-dir", default="outputs/sae_surprise", type=Path)
    args = ap.parse_args()

    surprise_csv = args.out_dir / f"candidates_layer_{args.layer}_surprise_vs__neutral.csv"
    afraid_csv   = args.out_dir / f"candidates_layer_{args.layer}_afraid_vs__neutral.csv"
    epistem_csv  = args.out_dir / f"epistemic_layer_{args.layer}.csv"
    for p in (surprise_csv, afraid_csv, epistem_csv):
        if not p.exists():
            raise SystemExit(f"Missing input: {p}")

    surprise = load_csv_indexed(surprise_csv)
    afraid   = load_csv_indexed(afraid_csv)
    epistem  = load_csv_indexed(epistem_csv)

    # Iterate over the surprise candidate set (the primary ranking).
    rows_out = []
    for fid, srow in surprise.items():
        ms = max(0.0, float(srow["mean_surprise"]))
        # afraid CSV has columns mean_afraid, mean__neutral, ...
        arow = afraid.get(fid)
        ma = max(0.0, float(arow["mean_afraid"])) if arow else 0.0
        specificity = ms / (ms + ma + EPS)

        erow = epistem.get(fid)
        if erow is None:
            verdict = "other"
            extras = {"paired_t": "", "subcat_coverage": "", "epistemic_verdict": "missing"}
        else:
            paired_t = float(erow["paired_t"])
            cov = int(erow["subcat_coverage"])
            extras = {"paired_t": paired_t, "subcat_coverage": cov,
                      "epistemic_verdict": erow["verdict"]}
            if cov >= 4 and paired_t > 2.5:
                verdict = "epistemic"
            elif specificity > 0.70:
                verdict = "surprise_affective"
            elif 0.40 <= specificity <= 0.60:
                verdict = "conflate"
            else:
                verdict = "other"

        rows_out.append({
            "feature_id": fid,
            "delta_surprise_vs_neutral": float(srow["delta"]),
            "mean_surprise": ms,
            "mean_afraid": ma,
            "mean_neutral_from_s": float(srow["mean__neutral"]),
            "specificity": specificity,
            **extras,
            "verdict": verdict,
        })

    out = args.out_dir / f"final_layer_{args.layer}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"Wrote {out}")

    # Quick stdout summary.
    by_verdict: dict[str, int] = {}
    for r in rows_out:
        by_verdict[r["verdict"]] = by_verdict.get(r["verdict"], 0) + 1
    print("Verdict counts:", by_verdict)
    epistemic_top = [r for r in rows_out if r["verdict"] == "epistemic"][:5]
    if epistemic_top:
        print("\nTop epistemic candidates:")
        for r in epistemic_top:
            print(f"  feat {r['feature_id']:>6}: paired_t={r['paired_t']:.2f}  "
                  f"coverage={r['subcat_coverage']}  spec={r['specificity']:.2f}  "
                  f"delta={r['delta_surprise_vs_neutral']:.3f}")


if __name__ == "__main__":
    main()
