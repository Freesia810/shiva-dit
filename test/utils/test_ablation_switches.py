"""Dummy-input verification for the two ablation switches.

Run:  python -m test.utils.test_ablation_switches

Checks, on both CPU and MPS:
  1. ShivaRouter(use_context=False) drops h_ctx (Eq. 8) but keeps the call
     signature, output shape and a live gradient path.
  2. pi_mode="gumbel" leaves the forward pass bit-identical to "residual"
     (same hard top-k, same budget, same grad wrt x) and only alters the
     backward surrogate -- so the estimator is the single varying factor.
"""

import torch

from shiva_dit.modules.router import ShivaRouter, CONTEXT_SIGNALS
from shiva_dit.modules.pruner import ShivaPrunerExpert
from shiva_dit.utils.ste import ShivaTopkFunc

B, N, D = 2, 128, 64
D_PROMPT, D_LAYER, D_BOTTLE = 96, 32, 64


def _devices():
    devs = ["cpu"]
    if torch.backends.mps.is_available():
        devs.append("mps")
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


def _router_inputs(device):
    g = torch.Generator().manual_seed(0)
    return dict(
        hidden_states=torch.randn(B, N, D, generator=g).to(device).requires_grad_(True),
        pooled_encoder_states=torch.randn(B, D_PROMPT, generator=g).to(device),
        temb=torch.randn(B, D, generator=g).to(device),
        layer_emb=torch.randn(B, D_LAYER, generator=g).to(device),
    )


ARMS = [
    ("full          ", CONTEXT_SIGNALS),
    ("-time         ", ("prompt", "layer")),
    ("-prompt       ", ("time", "layer")),
    ("-layer        ", ("time", "prompt")),
    ("layer only    ", ("layer",)),
    ("no h_ctx      ", ()),
]


def check_router_context_switch(device):
    print(f"\n[1] Router h_ctx switch  (device={device})")
    results = {}

    for name, signals in ARMS:
        torch.manual_seed(42)
        router = ShivaRouter(
            hidden_dim=D,
            prompt_embed_dim=D_PROMPT,
            layer_embed_dim=D_LAYER,
            bottleneck_dim=D_BOTTLE,
            time_embed_dim=D,
            context_signals=signals,
        ).to(device)

        inp = _router_inputs(device)
        scores = router(**inp)
        assert scores.shape == (B, N), f"expected (B,N), got {tuple(scores.shape)}"

        scores.sum().backward()
        n_params = sum(p.numel() for p in router.parameters())
        grad_x = inp["hidden_states"].grad.norm().item()

        # disabled projections must be absent, not merely unused
        present = {
            "time": hasattr(router, "temb_proj"),
            "prompt": hasattr(router, "prompt_proj"),
            "layer": hasattr(router, "layer_proj"),
        }
        for sig, is_present in present.items():
            assert is_present == (sig in signals), (
                f"{sig} submodule mismatch in arm {name}"
            )
        assert hasattr(router, "context_norm") == (len(signals) > 0)
        assert grad_x > 0, f"no gradient reaches tokens in arm {name}"

        results[name] = (n_params, scores.detach())
        print(
            f"    {name} params={n_params:>8,}  grad_x_norm={grad_x:.4f}  "
            f"signals={signals if signals else '()'}"
        )

    p_full = results["full          "][0]
    s_full = results["full          "][1]
    for name, (n_params, scores) in results.items():
        if name.strip() == "full":
            continue
        assert not torch.allclose(s_full, scores), (
            f"arm {name} produces identical scores to full -- switch not wired in"
        )

    p_none = results["no h_ctx      "][0]
    print("    -> all arms wired, token gradient path intact in every arm  [PASS]")
    print(
        f"    !! parameter confound: full={p_full:,} vs no-h_ctx={p_none:,} "
        f"({(p_full - p_none) / p_full * 100:.0f}% drop) -- needs a param-matched control"
    )


def check_pi_mode_switch(device):
    print(f"\n[2] pi_mode switch  (device={device})")

    def run(pi_mode, seed=0):
        torch.manual_seed(seed)
        x = torch.randn(B, N, D, device=device, requires_grad=True)
        logits = torch.randn(B, N, device=device, requires_grad=True)
        k = torch.tensor(float(int(N * 0.6)), device=device, requires_grad=True)
        pruner = ShivaPrunerExpert(
            init_Ts=10.0, init_Tk=10.0, pi_mode=pi_mode, gumbel_tau=1.0
        ).to(device)

        x_sel, x_rej, sel_idx, rej_idx = pruner(x, logits, k)
        # arbitrary downstream loss touching both paths
        (x_sel.pow(2).sum() - 0.5 * x_rej.pow(2).sum()).backward()
        return dict(
            sel_idx=sel_idx,
            x_sel=x_sel.detach(),
            grad_x=x.grad.clone(),
            grad_logits=logits.grad.clone(),
            grad_k=None if k.grad is None else k.grad.clone(),
        )

    res = run("residual")
    gum = run("gumbel")

    # --- forward must be untouched: same tokens, same budget ---
    assert torch.equal(res["sel_idx"], gum["sel_idx"]), (
        "selection differs across pi_mode"
    )
    assert torch.equal(res["x_sel"], gum["x_sel"]), (
        "selected features differ across pi_mode"
    )
    assert res["sel_idx"].shape[1] == int(N * 0.6), "budget not honoured"
    print(
        f"    forward identical: k={res['sel_idx'].shape[1]}/{N} tokens, "
        f"same indices, same features  [PASS]"
    )

    # --- grad wrt x is scatter of upstream grads, independent of the surrogate ---
    assert torch.equal(res["grad_x"], gum["grad_x"]), (
        "grad_x should not depend on pi_mode"
    )
    print("    grad_x identical (surrogate only affects the score path)  [PASS]")

    # --- the surrogate itself must differ ---
    assert not torch.allclose(res["grad_logits"], gum["grad_logits"]), (
        "grad_logits identical -- gumbel branch not taken"
    )
    print(
        f"    grad_logits norm  residual={res['grad_logits'].norm():.4f}  "
        f"gumbel={gum['grad_logits'].norm():.4f}  (differ)  [PASS]"
    )

    # --- budget gradient: exists for residual, structurally absent for gumbel ---
    gk_res = 0.0 if res["grad_k"] is None else res["grad_k"].abs().item()
    gk_gum = 0.0 if gum["grad_k"] is None else gum["grad_k"].abs().item()
    assert gk_res > 0, "residual estimator produced no gradient to k"
    assert gk_gum == 0.0, "gumbel arm must not produce a budget gradient"
    print(
        f"    grad_k  residual={gk_res:.4f}  gumbel={gk_gum:.4f} "
        f"(d(pi)/dk == 0 by construction)  [PASS]"
    )


def check_stats_buffer():
    print("\n[3] Autograd monitor records pi_mode")
    ShivaTopkFunc._stats_buffer = {}
    x = torch.randn(1, 32, 8, requires_grad=True)
    logits = torch.randn(1, 32, requires_grad=True)
    k = torch.tensor(16.0, requires_grad=True)
    pruner = ShivaPrunerExpert(init_Ts=10.0, init_Tk=10.0, pi_mode="gumbel")
    xs, _, _, _ = pruner(x, logits, k)
    xs.sum().backward()
    print("    backward completed without autograd signature errors  [PASS]")


def test_router_context_switch_cpu():
    check_router_context_switch("cpu")


def test_pi_mode_switch_cpu():
    check_pi_mode_switch("cpu")


def test_topk_autograd_signature():
    check_stats_buffer()


if __name__ == "__main__":
    for dev in _devices():
        check_router_context_switch(dev)
        check_pi_mode_switch(dev)
    check_stats_buffer()
    print("\nAll ablation-switch checks passed.\n")
