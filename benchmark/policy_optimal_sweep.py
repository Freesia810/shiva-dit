"""Measure the optimal retention schedule directly, without training a policy.

The additive-vs-FiLM ablation asks whether the optimal schedule needs a layer-time
interaction term. Training two policies answers it only indirectly: the comparison is
mediated by the budget controller, the sampling noise and the backbone adaptation, which
is why the short run in `ratio_coupling_ablation.py` measured which parameterisation
converges the budget faster rather than which one fits the optimum.

Here the optimum is measured instead. For every (layer, timestep) cell and a grid of
retention ratios we evaluate the Stage-1 objective on a captured block, giving a cost
surface C(layer, timestep, r). Allocating a global budget across cells by equalising the
marginal cost yields the optimal schedule r*(layer, timestep), which is constrained by no
parameterisation at all. Decomposing logit(r*) into a layer term plus a timestep term
then leaves a residual that is exactly what an additive policy cannot express.

    python -m benchmark.policy_optimal_sweep --caps /root/autodl-tmp/cap2 \
        --ckpt /root/autodl-tmp/assets/models/shiva-sd3-60.pt --out /root/autodl-tmp/cost.json
"""

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

import torch

from diffusers.models.attention import JointTransformerBlock
from shiva_dit.modules.router import ShivaRouter

NAME = re.compile(r"L(\d+)_S(\d+)_P(\d+)\.pt$")
PREFIX = "base_model.model.transformer_blocks.{l}._hook."


def build_router(model_sd, layer, device, dtype):
    """Rebuild the router serving this layer, inferring dims from the weights.

    Routers are shared pairwise (G=12 over 24 layers), so a layer without its own
    weights is served by the one below it.
    """
    owner = layer
    p = PREFIX.format(l=owner) + "router."
    if not any(k.startswith(p) for k in model_sd):
        owner = layer - 1
        p = PREFIX.format(l=owner) + "router."
    w = {k[len(p) :]: v for k, v in model_sd.items() if k.startswith(p)}
    if not w:
        return None, None
    bottleneck, hidden = w["token_proj.weight"].shape
    router = ShivaRouter(
        hidden_dim=hidden,
        prompt_embed_dim=w["prompt_proj.weight"].shape[1],
        layer_embed_dim=w["layer_proj.weight"].shape[1],
        bottleneck_dim=bottleneck,
        time_embed_dim=w["temb_proj.weight"].shape[1],
    )
    router.load_state_dict({k: v.float() for k, v in w.items()})
    # the layer-index embedding table is stored once, on the first hooked block
    lek = next(k for k in model_sd if "router_layer_embedder.weight" in k)
    layer_emb_w = model_sd[lek].float()
    return router.to(device, dtype).eval(), layer_emb_w.to(device, dtype)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--caps", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    model_sd = torch.load(args.ckpt, map_location="cpu")["model"]
    files = sorted(glob.glob(os.path.join(args.caps, "L*_S*_P*.pt")))
    print(f"{len(files)} captures, ratios {args.ratios}")

    blocks, routers = {}, {}
    # cost[(layer, step)][ratio] -> list of per-prompt losses
    cost = defaultdict(lambda: defaultdict(list))
    raw = defaultdict(lambda: defaultdict(list))

    for i, f in enumerate(files):
        m = NAME.search(os.path.basename(f))
        layer, step = int(m.group(1)), int(m.group(2))
        d = torch.load(f, map_location="cpu")
        x = d["hidden_states"].to(device, dtype)
        ctx = d["encoder_hidden_states"].to(device, dtype)
        temb = d["temb"].to(device, dtype)
        if temb.dim() == 3:
            temb = temb.squeeze(1)

        if layer not in blocks:
            wp = os.path.join(args.caps, f"weights_L{layer:02d}.pt")
            sd = torch.load(wp, map_location="cpu")
            cfg = d["block_config"]
            blk = JointTransformerBlock(
                dim=cfg["hidden_dim"],
                num_attention_heads=cfg["num_attention_heads"],
                attention_head_dim=cfg["attention_head_dim"],
                context_pre_only=cfg.get("context_pre_only", False),
            )
            blk.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
            blocks[layer] = blk.to(device, dtype).eval()
            routers[layer] = build_router(model_sd, layer, device, dtype)

        blk = blocks[layer]
        router, layer_emb_w = routers[layer]
        if router is None:
            continue

        with torch.no_grad():
            y_full = blk(hidden_states=x, encoder_hidden_states=ctx, temb=temb)[1]
            scores = router(
                x, ctx.mean(dim=1), temb, layer_emb_w[layer].unsqueeze(0)
            ).squeeze(-1)
            N = x.shape[1]
            for r in args.ratios:
                k = max(8, min(int(math.ceil(r * N)), N))
                idx = torch.topk(scores, k, dim=1).indices
                gid = idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])
                out = blk(
                    hidden_states=torch.gather(x, 1, gid),
                    encoder_hidden_states=ctx,
                    temb=temb,
                )[1]
                # rejected tokens keep their input through the identity skip
                # Error is scored on the retained positions only. A dropped token keeps
                # its input, so its error is exactly the update the block would have
                # applied to it, and summing those over dropped tokens is linear in how
                # many were dropped. That accounting term swamps the quantity the
                # schedule actually trades off, which is how much the surviving tokens
                # degrade once attention sees a shorter context.
                kept = torch.gather(y_full, 1, gid).float()
                yn = torch.nn.functional.layer_norm(out.float(), (out.shape[-1],))
                fn = torch.nn.functional.layer_norm(kept, (kept.shape[-1],))
                cost[(layer, step)][r].append(torch.mean((yn - fn) ** 2).item())
                # the all-token error, kept for reference
                y = x.clone().scatter(1, gid, out)
                raw[(layer, step)][r].append(
                    torch.mean((y.float() - y_full.float()) ** 2).item()
                )

        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(files)}", flush=True)

    out = {
        "ratios": args.ratios,
        "cost": {
            f"{layer_idx}_{step_idx}": {
                str(r): sum(v) / len(v) for r, v in values.items()
            }
            for (layer_idx, step_idx), values in cost.items()
        },
        "cost_raw": {
            f"{layer_idx}_{step_idx}": {
                str(r): sum(v) / len(v) for r, v in values.items()
            }
            for (layer_idx, step_idx), values in raw.items()
        },
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {args.out}  ({len(out['cost'])} cells)")


if __name__ == "__main__":
    main()
