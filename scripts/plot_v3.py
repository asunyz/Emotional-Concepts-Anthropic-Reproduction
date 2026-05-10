"""
Cognitive v3 — All four primary figures from extracted vectors.

Reads `concept_vectors_modeA.npz` and `trajectory_vectors_modeB.npz` produced
by `extract_v3.py` and writes:

  outputs/cognitive_v3_<task>/
    01_cosine_modeA.png       — 9x9 cosine matrix per layer (stage-block view)
    02_pca_modeA.png          — PCA scatter of Mode A vectors, color by stage
    03_layer_scan.png         — within-stage vs between-stage cosine across layers
    04_arithmetic.png         — vector arithmetic table per trajectory

  outputs/cognitive_v3_<task>/
    05_pca_modeB.png          — PCA of trajectory vectors, color by reaction

Usage:
    python scripts/plot_v3.py --run-dir runs/cognitive_v3_sanity --layers 10,20,30,36

Notes for sanity-scale data: with n=1-4 stories per stage-concept, all of these
figures are noisy and should be read as a smoke test of the extraction +
plotting pipeline, NOT as a scientific result. Re-run after the full data
collection finishes for trustworthy figures.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Stage / color conventions
# ---------------------------------------------------------------------------

STAGE_OF = {
    "curious": "P1", "uncertain": "P1", "confident": "P1",
    "surprised": "P2", "bored": "P2",
    "stubborn": "P3", "enlightened": "P3", "confused": "P3", "confirmed": "P3",
}

# Concept order chosen to make stage blocks visible in the cosine matrix:
# Prior block (3) → Discovery block (2) → Reaction block (4).
CONCEPT_ORDER = [
    "curious", "uncertain", "confident",
    "surprised", "bored",
    "stubborn", "enlightened", "confused", "confirmed",
]

STAGE_COLOR = {"P1": "#1f77b4", "P2": "#ff7f0e", "P3": "#2ca02c"}

REACTION_COLOR = {
    "stubborn":    "#d62728",
    "enlightened": "#2ca02c",
    "confused":    "#9467bd",
    "confirmed":   "#1f77b4",
}


def normed(V: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return V / (np.linalg.norm(V, axis=-1, keepdims=True) + eps)


def cosine_matrix(V: np.ndarray) -> np.ndarray:
    Vn = normed(V)
    return Vn @ Vn.T


def load_modeA(layer_dir: Path) -> tuple[list[str], np.ndarray]:
    """Return (concept_names, V) — concept_names in CONCEPT_ORDER (filtered to those present)."""
    npz = np.load(layer_dir / "concept_vectors_modeA.npz")
    available = set(npz.files)
    names = [c for c in CONCEPT_ORDER if c in available]
    V = np.stack([npz[c] for c in names])
    return names, V


def load_modeB(layer_dir: Path) -> tuple[list[str], np.ndarray, list[str]]:
    """Return (trajectory_names, V, reactions)."""
    npz = np.load(layer_dir / "trajectory_vectors_modeB.npz")
    names = sorted(npz.files)
    V = np.stack([npz[n] for n in names])
    reactions = [n.split("-")[2] for n in names]  # "traj_01_confident-surprised-stubborn" -> "stubborn"
    return names, V, reactions


# ---------------------------------------------------------------------------
# Figure 1: cosine matrix per layer (Mode A)
# ---------------------------------------------------------------------------

def fig_cosine(out_path: Path, run_dir: Path, layers: list[int]) -> None:
    fig, axes = plt.subplots(1, len(layers), figsize=(4.6 * len(layers), 4.6),
                             squeeze=False)
    for ax, L in zip(axes[0], layers):
        names, V = load_modeA(run_dir / f"layer_{L}")
        S = cosine_matrix(V)
        im = ax.imshow(S, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.set_title(f"layer {L}")
        # stage block separators
        n_p1 = sum(1 for n in names if STAGE_OF[n] == "P1")
        n_p2 = sum(1 for n in names if STAGE_OF[n] == "P2")
        for sep in [n_p1 - 0.5, n_p1 + n_p2 - 0.5]:
            ax.axhline(sep, color="black", lw=0.8)
            ax.axvline(sep, color="black", lw=0.8)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{S[i, j]:+.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if abs(S[i, j]) > 0.5 else "black")
        plt.colorbar(im, ax=ax, fraction=0.045)
    fig.suptitle("Mode A — cosine similarity (stage blocks: prior | discovery | reaction)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: PCA scatter of Mode A vectors per layer (color by stage)
# ---------------------------------------------------------------------------

def fig_pca_modeA(out_path: Path, run_dir: Path, layers: list[int]) -> None:
    fig, axes = plt.subplots(1, len(layers), figsize=(4.8 * len(layers), 4.8),
                             squeeze=False)
    for ax, L in zip(axes[0], layers):
        names, V = load_modeA(run_dir / f"layer_{L}")
        Z = PCA(n_components=2).fit_transform(V)
        for i, n in enumerate(names):
            stage = STAGE_OF[n]
            ax.scatter(Z[i, 0], Z[i, 1], s=130, c=STAGE_COLOR[stage],
                       edgecolors="black", linewidths=0.6,
                       label=stage if (stage not in {a.get_label() for a in ax.collections[:i]}) else None)
            ax.annotate(n, Z[i], fontsize=9, xytext=(5, 5), textcoords="offset points")
        ax.axhline(0, color="gray", lw=0.4); ax.axvline(0, color="gray", lw=0.4)
        ax.set_title(f"layer {L}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    # one legend
    handles = [plt.Line2D([], [], marker='o', linestyle='', color=c, markersize=10,
                          markeredgecolor='black', label=stage)
               for stage, c in STAGE_COLOR.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Mode A — PCA of stage-wise concept vectors", fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: layer scan — within-stage vs between-stage cosine
# ---------------------------------------------------------------------------

def fig_layer_scan(out_path: Path, run_dir: Path, layers: list[int]) -> None:
    """For each layer, plot mean within-stage cosine and mean between-stage
    cosine. A working stage-block representation should show within > between
    that grows with depth."""
    within: list[float] = []
    between: list[float] = []
    for L in layers:
        names, V = load_modeA(run_dir / f"layer_{L}")
        S = cosine_matrix(V)
        n = len(names)
        w_vals, b_vals = [], []
        for i in range(n):
            for j in range(i + 1, n):
                if STAGE_OF[names[i]] == STAGE_OF[names[j]]:
                    w_vals.append(S[i, j])
                else:
                    b_vals.append(S[i, j])
        within.append(np.mean(w_vals) if w_vals else np.nan)
        between.append(np.mean(b_vals) if b_vals else np.nan)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(layers, within, "o-", color="#2ca02c", label="within-stage (avg)", linewidth=2)
    ax.plot(layers, between, "s-", color="#d62728", label="between-stage (avg)", linewidth=2)
    ax.axhline(0, color="gray", lw=0.4)
    ax.set_xlabel("layer"); ax.set_ylabel("mean cosine")
    ax.set_title("Layer scan — does stage-block separation grow with depth?")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: vector arithmetic — v_prior + v_discovery vs v_reaction
# ---------------------------------------------------------------------------

def fig_arithmetic(out_path: Path, run_dir: Path, layers: list[int],
                   trajectories: list[dict]) -> None:
    """For each (layer, trajectory), compute cos(v_prior + v_discovery, v_reaction).
    Mode A vectors are used. A 'working' compositionality result would have
    most cosines > 0.3 and far above the random baseline (which we also plot).
    """
    fig, ax = plt.subplots(figsize=(max(7, 1.0 * len(trajectories) + 2), 4.5))
    bar_w = 0.18
    x = np.arange(len(trajectories))

    rng = np.random.default_rng(0)
    for k, L in enumerate(layers):
        names, V = load_modeA(run_dir / f"layer_{L}")
        n2v = dict(zip(names, V))
        cosines = []
        for traj in trajectories:
            pri, dis, rea = traj["prior"], traj["discovery"], traj["reaction"]
            if pri not in n2v or dis not in n2v or rea not in n2v:
                cosines.append(np.nan)
                continue
            lhs = n2v[pri] + n2v[dis]
            rhs = n2v[rea]
            num = float(lhs @ rhs)
            den = (np.linalg.norm(lhs) * np.linalg.norm(rhs)) + 1e-9
            cosines.append(num / den)

        # random baseline: cos with a random-direction vector of the same norm
        rand_baseline = []
        for _ in range(50):
            r = rng.standard_normal(V.shape[1])
            r /= np.linalg.norm(r)
            ref = V[0] / (np.linalg.norm(V[0]) + 1e-9)
            rand_baseline.append(float(r @ ref))
        rand_mean = np.mean(rand_baseline); rand_std = np.std(rand_baseline)

        ax.bar(x + (k - (len(layers) - 1) / 2) * bar_w, cosines, bar_w,
               label=f"L={L}")
        if k == 0:
            ax.axhspan(rand_mean - rand_std, rand_mean + rand_std,
                       color="gray", alpha=0.15, label="random baseline (±1σ)")

    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x, [f"#{t['id']}\n{t['prior'][:4]}+{t['discovery'][:4]}\n→{t['reaction'][:5]}"
                      for t in trajectories], fontsize=7)
    ax.set_ylabel("cos(v_prior + v_discovery, v_reaction)")
    ax.set_title("Vector arithmetic — Bayesian flow compositionality")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 5: PCA of Mode B trajectory vectors, colored by reaction
# ---------------------------------------------------------------------------

def fig_pca_modeB(out_path: Path, run_dir: Path, layers: list[int]) -> None:
    fig, axes = plt.subplots(1, len(layers), figsize=(4.8 * len(layers), 4.8),
                             squeeze=False)
    for ax, L in zip(axes[0], layers):
        names, V, reactions = load_modeB(run_dir / f"layer_{L}")
        Z = PCA(n_components=2).fit_transform(V)
        for i, (n, r) in enumerate(zip(names, reactions)):
            ax.scatter(Z[i, 0], Z[i, 1], s=140,
                       c=REACTION_COLOR.get(r, "gray"),
                       edgecolors="black", linewidths=0.6)
            short = n.split("_", 2)[2] if "_" in n else n
            ax.annotate(short, Z[i], fontsize=7, xytext=(5, 5),
                        textcoords="offset points")
        ax.axhline(0, color="gray", lw=0.4); ax.axvline(0, color="gray", lw=0.4)
        ax.set_title(f"layer {L}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    handles = [plt.Line2D([], [], marker='o', linestyle='', color=c, markersize=10,
                          markeredgecolor='black', label=r)
               for r, c in REACTION_COLOR.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Mode B — PCA of trajectory vectors (color by reaction)", fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--layers", default="10,20,30,36")
    ap.add_argument("--output-dir", default=None,
                    help="defaults to outputs/<run-dir-name>/")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    layers = [int(x) for x in args.layers.split(",")]
    if args.output_dir is None:
        out_dir = Path("outputs") / run_dir.name
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_cfg = json.loads((run_dir / "trajectories.json").read_text())
    trajectories = traj_cfg["trajectories"]

    fig_cosine(out_dir / "01_cosine_modeA.png", run_dir, layers)
    fig_pca_modeA(out_dir / "02_pca_modeA.png", run_dir, layers)
    fig_layer_scan(out_dir / "03_layer_scan.png", run_dir, layers)
    fig_arithmetic(out_dir / "04_arithmetic.png", run_dir, layers, trajectories)
    # Mode B is optional — only methods that produce trajectory_vectors_modeB.npz
    if (run_dir / f"layer_{layers[0]}" / "trajectory_vectors_modeB.npz").exists():
        fig_pca_modeB(out_dir / "05_pca_modeB.png", run_dir, layers)
    else:
        print("(skipping 05_pca_modeB.png — no trajectory_vectors_modeB.npz)")

    print(f"All figures in {out_dir}")


if __name__ == "__main__":
    main()
