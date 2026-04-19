"""
Concept-vector extraction pipeline.

For each concept c and each layer L we produce a vector that points in the
direction of c after (a) subtracting the mean over all concepts and
(b) projecting off the top principal components of a neutral corpus (those
that explain ≥50% of the neutral variance — i.e. "task-generic" directions).

Layout:  <output-dir>/<task-label>/
    prompts.json                              # provenance
    stories/<concept>-<topic_idx>-<story_idx>.txt
    stories/_neutral-<topic_idx>-<story_idx>.txt
    layer_<L>/
        raw_concept/<concept>-<tidx>-<sidx>.npy
        raw_neutral/<tidx>-<sidx>.npy
        concept_vectors.npz                   # one array per concept
        mean.npy                              # mean over concepts
        neutral_projection.npy                # [d, k] orthonormal basis

Resuming: any file that already exists on disk is reused. Remove specific
files to force regeneration.

Usage:
python extract_concepts.py \
    --concept-prompt inputs/emotions/concept_prompt.txt \
    --concept-topics inputs/emotions/concept_topics.txt \
    --concepts inputs/emotions/concepts.csv \
    --neutral-prompt inputs/emotions/neutral_prompt.txt \
    --neutral-topics inputs/emotions/concept_topics.txt \
    --layers 16,24 \
    --task-label emotions_8b_nf4
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from cv_utils import load_model, extract_layer_activations, generate_story

AVG_FROM_TOKEN = 50

# Matches a whole line whose only "content" is a story/dialogue tag.
# Tolerates any surrounding punctuation/markdown: [story 1], **story 1**,
# ### Story 1 ###, story__1, (dialogue-2), etc. The line must contain
# nothing else — a sentence that happens to mention "story 1" won't match.
SPLIT_RE = re.compile(r"^[\W_]*(?:story|dialogue)[\W_]*\d+[\W_]*$",
                      re.I | re.M)

# Line that opens a new turn in a Person/AI dialogue.
TURN_RE = re.compile(r"^\s*(Person|AI)\s*:\s*(.*)$", re.I)


def split_stories(text: str, n: int) -> list[str]:
    parts = [p.strip() for p in SPLIT_RE.split(text) if p.strip()]
    return parts[:n]


def parse_dialogue(text: str) -> list[dict]:
    """Parse a Person/AI transcript into chat-template messages.

    Anything before the first 'Person:' / 'AI:' line becomes a system message.
    Returns [] if no turns are found (caller should fall back to raw text).
    """
    messages: list[dict] = []
    system_lines: list[str] = []
    turns: list[tuple[str, list[str]]] = []  # (role, buffered lines)

    for line in text.splitlines():
        m = TURN_RE.match(line)
        if m:
            role = "user" if m.group(1).lower() == "person" else "assistant"
            turns.append((role, [m.group(2)] if m.group(2).strip() else []))
        elif not turns:
            system_lines.append(line)
        else:
            turns[-1][1].append(line)

    system_text = "\n".join(system_lines).strip()
    if system_text:
        messages.append({"role": "system", "content": system_text})
    for role, buf in turns:
        content = "\n".join(buf).strip()
        if content:
            messages.append({"role": role, "content": content})
    return messages


def dialogue_to_chat_text(text: str, tokenizer) -> str:
    """Render a dialogue through the model's chat template for a context-free embedding pass."""
    messages = parse_dialogue(text)
    if not messages:
        return text
    return tokenizer.apply_chat_template(messages, tokenize=False)


def read_csv_file(p: str | Path) -> list[str]:
    return [x.strip() for x in Path(p).read_text().split(",") if x.strip()]


def read_lines(p: str | Path) -> list[str]:
    return [l.strip() for l in Path(p).read_text().splitlines() if l.strip()]


def ensure_multi_story(model, stories_dir: Path, prefix: str, n: int,
                       prompt: str, max_new_tokens: int, temperature: float) -> None:
    """Generate N sub-stories for one (concept, topic) batch and write them to disk.

    Resume rule: if all N files `{prefix}-{s}.txt` already exist, do nothing.
    Otherwise regenerate the batch and overwrite the set (sub-stories are
    coupled to one sampled completion).
    """
    targets = [stories_dir / f"{prefix}-{s}.txt" for s in range(n)]
    if all(p.exists() for p in targets):
        return

    full = generate_story(model, prompt, max_new_tokens=max_new_tokens, temperature=temperature)
    # Keep the raw completion for provenance / debugging parser issues.
    raw_dir = stories_dir / "_raw"
    raw_dir.mkdir(exist_ok=True)
    (raw_dir / f"{prefix}.txt").write_text(full)

    substories = split_stories(full, n)
    if len(substories) < n:
        print(f"  WARN: parsed only {len(substories)}/{n} stories from '{prefix}'. "
              f"Check _raw/{prefix}.txt; consider raising --max-new-tokens or "
              f"adjusting the prompt's format instructions.")
    for path, text in zip(targets, substories):
        path.write_text(text)
    # stale leftovers when the new parse yielded fewer stories than before
    for path in targets[len(substories):]:
        if path.exists():
            path.unlink()


def ensure_raw_vectors(model, text: str, layers: list[int],
                       layer_dirs: dict[int, Path], subdir: str, name: str) -> None:
    """Compute+save per-layer averaged activation for `text`, skipping layers already on disk."""
    targets = {L: layer_dirs[L] / subdir / f"{name}.npy" for L in layers}
    missing = [L for L, p in targets.items() if not p.exists()]
    if not missing:
        return
    acts = extract_layer_activations(model, text, missing)
    for L, h in acts.items():
        if h.shape[0] <= AVG_FROM_TOKEN:
            raise ValueError(f"'{name}' has only {h.shape[0]} tokens (need > {AVG_FROM_TOKEN})")
        v = h[AVG_FROM_TOKEN:].mean(0).numpy()
        targets[L].parent.mkdir(parents=True, exist_ok=True)
        np.save(targets[L], v)


def load_raws(layer_dir: Path, subdir: str) -> tuple[list[str], np.ndarray]:
    files = sorted((layer_dir / subdir).glob("*.npy"))
    names = [f.stem for f in files]
    vecs = np.stack([np.load(f) for f in files]) if files else np.empty((0, 0))
    return names, vecs


def pca_variance_basis(X: np.ndarray, variance_fraction: float = 0.5) -> np.ndarray:
    """Return [d, k] orthonormal basis of top PCs explaining ≥ `variance_fraction`."""
    pca = PCA().fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cum, variance_fraction) + 1)
    return pca.components_[:k].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept-prompt", required=True,
                    help="text file with {concept}, {topic}, {n_stories} placeholders")
    ap.add_argument("--concept-topics", required=True, help="one topic per line")
    ap.add_argument("--concepts", required=True, help="single file, comma-separated concept words")
    ap.add_argument("--neutral-prompt", required=True,
                    help="text file with {topic}, {n_stories} placeholders")
    ap.add_argument("--neutral-topics", required=True, help="one topic per line")
    ap.add_argument("--layers", required=True, help="comma-separated layer indices, e.g. 12,16,20")
    ap.add_argument("--task-label", required=True)
    ap.add_argument("--output-dir", default="runs")
    ap.add_argument("--n-stories", type=int, default=5,
                    help="stories generated per (concept, topic) in a single completion")
    ap.add_argument("--max-new-tokens", type=int, default=1500,
                    help="budget for the whole multi-story completion (not per story)")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--model-path", default=None)
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    concepts = read_csv_file(args.concepts)
    concept_topics = read_lines(args.concept_topics)
    neutral_topics = read_lines(args.neutral_topics)
    concept_prompt = Path(args.concept_prompt).read_text()
    neutral_prompt = Path(args.neutral_prompt).read_text()

    root = Path(args.output_dir) / args.task_label
    stories_dir = root / "stories"
    stories_dir.mkdir(parents=True, exist_ok=True)
    layer_dirs = {L: root / f"layer_{L}" for L in layers}
    for d in layer_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    (root / "prompts.json").write_text(json.dumps({
        "concept_prompt": concept_prompt, "neutral_prompt": neutral_prompt,
        "concepts": concepts, "concept_topics": concept_topics,
        "neutral_topics": neutral_topics, "layers": layers,
        "n_stories": args.n_stories, "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature, "avg_from_token": AVG_FROM_TOKEN,
    }, indent=2))

    model = load_model(args.model_path)

    # =====================================================================
    # Phase 1: GENERATE all stories and write them to disk.
    # =====================================================================
    concept_batches = [(c, t, tidx) for c in concepts for tidx, t in enumerate(concept_topics)]
    pbar = tqdm(concept_batches, desc="gen concept", unit="batch")
    for concept, topic, tidx in pbar:
        prefix = f"{concept}-{tidx}"
        pbar.set_postfix_str(prefix)
        prompt = concept_prompt.format(concept=concept, topic=topic,
                                       n_stories=args.n_stories)
        ensure_multi_story(model, stories_dir, prefix, args.n_stories,
                           prompt, args.max_new_tokens, args.temperature)

    pbar = tqdm(list(enumerate(neutral_topics)), desc="gen neutral", unit="batch")
    for tidx, topic in pbar:
        prefix = f"_neutral-{tidx}"
        pbar.set_postfix_str(prefix)
        prompt = neutral_prompt.format(topic=topic, n_stories=args.n_stories)
        ensure_multi_story(model, stories_dir, prefix, args.n_stories,
                           prompt, args.max_new_tokens, args.temperature)

    # =====================================================================
    # Phase 2: EXTRACT activations. Each story is fed back through the model
    # on its own (no prompt, no sibling stories) so the embedding reflects
    # only the story text, not the generation context that produced it.
    # =====================================================================
    def story_files(pattern: str) -> list[Path]:
        return sorted(p for p in stories_dir.glob(pattern) if p.parent == stories_dir)

    concept_files = [p for p in story_files("*.txt") if not p.name.startswith("_neutral-")]
    neutral_files = story_files("_neutral-*.txt")

    for path in tqdm(concept_files, desc="embed concept", unit="story"):
        text = path.read_text()
        ensure_raw_vectors(model, text, layers, layer_dirs, "raw_concept", path.stem)

    for path in tqdm(neutral_files, desc="embed neutral", unit="story"):
        dialogue = path.read_text()
        # Feed neutral dialogues back through the chat template so the model
        # sees them the same way it would in an actual chat. Mapping:
        # leading prose → system, Person → user, AI → assistant.
        text = dialogue_to_chat_text(dialogue, model.tokenizer)
        name = path.stem[len("_neutral-"):]
        ensure_raw_vectors(model, text, layers, layer_dirs, "raw_neutral", name)

    # ---- per-layer reduction ----
    for L in tqdm(layers, desc="reducing", unit="layer"):
        ldir = layer_dirs[L]
        c_names, c_raw = load_raws(ldir, "raw_concept")
        n_names, n_raw = load_raws(ldir, "raw_neutral")

        # group per-story raw vectors by concept (prefix before "-<tidx>-<sidx>")
        by_concept: dict[str, list[np.ndarray]] = {}
        for name, v in zip(c_names, c_raw):
            concept = name.rsplit("-", 2)[0]
            by_concept.setdefault(concept, []).append(v)
        per_concept = {c: np.stack(vs).mean(0) for c, vs in by_concept.items()}

        mean_all = np.stack(list(per_concept.values())).mean(0)
        basis = pca_variance_basis(n_raw, 0.5)  # [d, k]

        # final = (I - B Bᵀ) (v − mean_all)
        final = {c: (v - mean_all) - basis @ (basis.T @ (v - mean_all))
                 for c, v in per_concept.items()}

        np.savez(ldir / "concept_vectors.npz", **final)
        np.save(ldir / "mean.npy", mean_all)
        np.save(ldir / "neutral_projection.npy", basis)
        print(f"[layer {L}] concepts={len(final)}  neutral-PCA k={basis.shape[1]}")


if __name__ == "__main__":
    main()
