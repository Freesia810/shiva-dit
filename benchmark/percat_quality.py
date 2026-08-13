"""Where does pruning actually hurt? Per-category, paired, at the 60% budget.

Reviewer yQcg asks to *measure* the quality drop on dense text or crowded scenes at the
60% budget -- the paper only asserts it in the Limitations. MJHQ's categories give a
handle on that: `logo` is where prompts most often demand rendered text, and `people` is
where scenes get crowded, so if the spatial-redundancy assumption really weakens there,
those two categories should degrade more than the rest.

Metrics are paired per-image deltas rather than per-category FID. FID over ~200 images
per category is dominated by small-sample bias, and per-category reference stats do not
exist; a paired delta needs no reference at all and cancels prompt difficulty, since both
arms score the same prompt.

All three of the paper's per-image metrics are reported, because they fail differently:
CLIP scores text-image alignment, CLIP-IQA scores perceptual quality with no reference,
and ImageReward scores human preference. A limitation carried by one metric alone is
thin evidence; agreement across the three is what makes it hold up.

    python -m benchmark.percat_quality --a <vanilla s00 dir> --b <shiva60 s00 dir> \
        --labels Vanilla Shiva-60
"""

import argparse
import glob
import json
import os
import re
import statistics as st

import torch
from PIL import Image
from tqdm import tqdm

NAME_RE = re.compile(r"p(\d+)_s(\d+)\.(?:png|jpg|jpeg)$", re.IGNORECASE)


def load_prompts(jsonl):
    """global index -> (prompt, category); the index is the line number."""
    out = {}
    with open(jsonl) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            txt = d.get("text") or d.get("prompt")
            cat = d.get("file_name", "?/").split("/")[0]
            out[i] = (txt, cat)
    return out


def index_images(d):
    out = {}
    for ext in ("png", "jpg", "jpeg"):
        for p in glob.glob(os.path.join(d, f"p*_s*.{ext}")):
            m = NAME_RE.search(os.path.basename(p))
            if m:
                out[int(m.group(1))] = p
    return out


def _shim_transformers_for_blip():
    """ImageReward bundles a BLIP copy that imports helpers transformers has moved.

    `apply_chunking_to_forward` and `prune_linear_layer` now live in
    `transformers.pytorch_utils`, and `find_pruneable_heads_and_indices` was dropped
    entirely. Only head pruning uses the last one and BLIP never prunes heads, but the
    import happens at module load, so it has to resolve.
    """
    import torch
    import transformers.modeling_utils as mu

    try:
        import transformers.pytorch_utils as pu
    except ImportError:
        pu = None
    for name in ("apply_chunking_to_forward", "prune_linear_layer"):
        if not hasattr(mu, name) and pu is not None and hasattr(pu, name):
            setattr(mu, name, getattr(pu, name))
    if not hasattr(mu, "find_pruneable_heads_and_indices"):

        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned):
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned
            for head in heads:
                head = head - sum(1 for h in already_pruned if h < head)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            return heads, torch.arange(len(mask))[mask].long()

        mu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
    # ImageReward's package init pulls in ReFL, which imports StableDiffusionPipeline
    # from diffusers. The repo ships its own diffusers fork on the path and that fork
    # does not export it. ReFL is a training utility we never call, so a stub is enough.
    import sys
    from types import ModuleType

    sys.modules.setdefault("ImageReward.ReFL", ModuleType("ImageReward.ReFL"))
    # BLIP reads `additional_special_tokens_ids`, a tokenizer property newer
    # transformers dropped; it is just a lookup of the additional special tokens.
    from transformers import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "additional_special_tokens_ids"):
        PreTrainedTokenizerBase.additional_special_tokens_ids = property(
            lambda self: self.convert_tokens_to_ids(self.additional_special_tokens)
        )


def build_scorers(device, want):
    """Return {metric: fn(pil, text) -> float} for whichever models actually load.

    Each metric is loaded in its own try block: CLIP-IQA and ImageReward are the
    fragile ones (torchmetrics/transformers pairings, a 1.8GB checkpoint), and a
    failure there must not take down the CLIP score the original finding rests on.
    """
    import torchvision.transforms.functional as TF

    scorers = {}

    clip_path = "../openai-clip-vit-base-patch16"
    if not os.path.isdir(clip_path):
        clip_path = "openai/clip-vit-base-patch16"

    if "clip" in want:
        from benchmark.generation import DirectClipScore

        clip = DirectClipScore(clip_path, device)

        def _clip(pil, text):
            t = (TF.to_tensor(pil).to(device) * 255).to(torch.uint8).unsqueeze(0)
            return clip(t, [text]).item()

        scorers["CLIP"] = _clip

    if "iqa" in want:
        try:
            from torchmetrics.multimodal import CLIPImageQualityAssessment

            iqa = CLIPImageQualityAssessment(model_name_or_path=clip_path).to(device)

            def _iqa(pil, _text):
                out = iqa(TF.to_tensor(pil).to(device).unsqueeze(0))
                if isinstance(out, dict):
                    return torch.stack(list(out.values())).mean().item()
                return out.item()

            scorers["IQA"] = _iqa
        except Exception as e:
            print(f"[skip] CLIP-IQA unavailable: {e}")

    if "ir" in want:
        try:
            _shim_transformers_for_blip()
            import ImageReward as RM

            # the generation boxes keep the checkpoint beside the repo; a scoring-only
            # machine has no such directory, so fall back to the hub release
            local = "../ImageReward/ImageReward.pt"
            if os.path.exists(local):
                ir = RM.load(
                    local, device=device, med_config="../ImageReward/med_config.json"
                )
            else:
                ir = RM.load("ImageReward-v1.0", device=device)
            scorers["IR"] = lambda pil, text: float(ir.score(text, pil))
        except Exception as e:
            print(f"[skip] ImageReward unavailable: {e}")

    return scorers


def report(name, per_cat, labels):
    print(f"\n=== {name} ===")
    print(
        f"{'category':<12}{'n':>5}{'delta':>10}{'stderr':>9}{'sigma':>8}{'worse %':>9}"
    )
    print("-" * 53)
    summary = {}
    for cat, ds in sorted(per_cat.items(), key=lambda kv: st.mean(kv[1])):
        m = st.mean(ds)
        se = st.stdev(ds) / (len(ds) ** 0.5) if len(ds) > 1 else 0.0
        worse = 100.0 * sum(1 for x in ds if x < 0) / len(ds)
        sigma = m / se if se else 0.0
        summary[cat] = {
            "n": len(ds),
            "mean": m,
            "stderr": se,
            "worse_pct": worse,
            "sigma": sigma,
        }
        print(f"{cat:<12}{len(ds):>5}{m:>10.4f}{se:>9.4f}{sigma:>8.2f}{worse:>8.1f}%")

    allds = [x for ds in per_cat.values() for x in ds]
    m = st.mean(allds)
    se = st.stdev(allds) / len(allds) ** 0.5
    print("-" * 53)
    print(
        f"{'ALL':<12}{len(allds):>5}{m:>10.4f}{se:>9.4f}{m / se:>8.2f}"
        f"{100.0 * sum(1 for x in allds if x < 0) / len(allds):>8.1f}%"
    )
    summary["__all__"] = {"n": len(allds), "mean": m, "stderr": se}
    print(
        f"delta = {labels[1]} minus {labels[0]}; negative means the pruned arm "
        f"scores lower."
    )
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="baseline image dir")
    p.add_argument("--b", required=True, help="pruned image dir")
    p.add_argument("--labels", nargs=2, default=["A", "B"])
    p.add_argument("--jsonl", default="benchmark/prompts/eval.jsonl")
    p.add_argument("--out", default="results/percat_quality_multi.json")
    p.add_argument(
        "--metrics",
        nargs="+",
        default=["clip", "iqa", "ir"],
        choices=["clip", "iqa", "ir"],
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="score only the first N shared prompts (smoke test)",
    )
    args = p.parse_args()

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    scorers = build_scorers(device, set(args.metrics))
    if not scorers:
        raise SystemExit("no metric could be loaded")
    print(f"metrics: {', '.join(scorers)}   device={device}")

    prompts = load_prompts(args.jsonl)
    ia, ib = index_images(args.a), index_images(args.b)
    shared = sorted(set(ia) & set(ib) & set(prompts))
    if args.limit:
        shared = shared[: args.limit]
    print(f"{len(shared)} prompts scored in both arms")

    # per_cat[metric][category] -> list of paired deltas
    per_cat = {name: {} for name in scorers}
    for gid in tqdm(shared, desc="scoring"):
        text, cat = prompts[gid]
        if not text:
            continue
        # both arms opened once per prompt, then scored by every metric, so the
        # deltas all describe exactly the same pair of images
        pa = Image.open(ia[gid]).convert("RGB")
        pb = Image.open(ib[gid]).convert("RGB")
        for name, fn in scorers.items():
            try:
                per_cat[name].setdefault(cat, []).append(fn(pb, text) - fn(pa, text))
            except Exception as e:
                print(f"\n[warn] {name} failed on prompt {gid}: {e}")

    out = {"labels": args.labels, "n_prompts": len(shared), "metrics": {}}
    for name in scorers:
        if per_cat[name]:
            out["metrics"][name] = report(name, per_cat[name], args.labels)

    # agreement across metrics is the whole point of running all three
    if len(out["metrics"]) > 1:
        names = list(out["metrics"])
        cats = set.intersection(*(set(out["metrics"][n]) - {"__all__"} for n in names))
        ranks = {
            n: {
                c: i + 1
                for i, c in enumerate(
                    sorted(cats, key=lambda c: out["metrics"][n][c]["mean"])
                )
            }
            for n in names
        }
        print("\n=== rank of each category (1 = worst) across metrics ===")
        print(f"{'category':<12}" + "".join(f"{n:>8}" for n in names))
        print("-" * (12 + 8 * len(names)))
        for c in sorted(cats, key=lambda c: ranks[names[0]][c]):
            print(f"{c:<12}" + "".join(f"{ranks[n][c]:>8}" for n in names))
        out["ranks"] = ranks

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
