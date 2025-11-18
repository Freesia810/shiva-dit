import torch
import torch.nn as nn
from typing import Type

from shiva_dit.hook.hook_manager import HookManager
from shiva_dit.utils import global_context

from shiva_dit.modules.router import ShivaRouter
from shiva_dit.modules.gater import ShivaGaterPolicy, ShivaRatioPolicy
from shiva_dit.modules.pruner import ShivaContextPruner, ShivaPrunerExpert
from shiva_dit.modules.fuser import ShivaFuserExpert
from shiva_dit.shiva_hook import ShivaDitBlockHook


def install_shiva_hooks(
    hook_manager_cls: Type["HookManager"],
    model_type: str,
    pipe,
    model_attrs: list[str],
    hook_cls_to_install: Type[ShivaDitBlockHook],
    **kwargs,
):
    bottleneck_dim = kwargs.pop("bottleneck_dim", 64)
    num_router_groups = kwargs.pop("num_router_groups", 4)
    layer_embed_dim = kwargs.pop("layer_embed_dim", 256)
    fuse_expert_ffn_mult = kwargs.pop("fuse_expert_ffn_mult", 2)
    context_pruning_threshold = kwargs.pop("context_pruning_threshold", None)
    context_pruning_ratio = kwargs.pop("context_pruning_ratio", None)
    init_Ts = kwargs.pop("init_Ts", 30.0)
    init_Tk = kwargs.pop("init_Tk", 30.0)

    model_dim = global_context.hidden_dim
    prompt_embed_dim = global_context.encoder_hidden_dim
    if model_dim is None:
        raise ValueError("[ShivaInstaller] 'hidden_dim' not found in global_context.")
    if prompt_embed_dim is None:
        raise ValueError(
            "[ShivaInstaller] 'encoder_hidden_dim' not found in global_context."
        )

    time_embed_dim = model_dim

    all_blocks = []
    for model_attr in model_attrs:
        m = getattr(pipe, model_attr)
        for name, module in m.named_modules():
            if isinstance(module, hook_manager_cls.get_block_type(model_type)):
                full_name = f"{model_attr}.{name}"
                all_blocks.append((full_name, module))

    num_layers = len(all_blocks)
    if num_layers == 0:
        print("[ShivaInstaller] Warning: Found 0 matching blocks.")
        return

    try:
        first_block_param = next(all_blocks[0][1].parameters())
        device = first_block_param.device
        dtype = first_block_param.dtype
    except StopIteration:
        raise RuntimeError("Cannot infer device/dtype: First block has no parameters.")

    print(
        f"[ShivaInstaller] Found {num_layers} blocks. Instantiating shared modules..."
    )
    print(
        f"  > model_dim: {model_dim}, prompt_embed_dim: {prompt_embed_dim}, layer_embed_dim: {layer_embed_dim}"
    )

    shared_layer_embedder = nn.Embedding(num_layers, layer_embed_dim).to(device, dtype)
    shared_gater_policy = ShivaGaterPolicy(
        time_embed_dim=time_embed_dim, layer_embed_dim=layer_embed_dim
    ).to(device, dtype)
    shared_ratio_policy = ShivaRatioPolicy(
        time_embed_dim=time_embed_dim, layer_embed_dim=layer_embed_dim
    ).to(device, dtype)
    shared_context_pruner = ShivaContextPruner(
        threshold=context_pruning_threshold, ratio=context_pruning_ratio
    )
    shared_pruner_expert = ShivaPrunerExpert(init_Ts=init_Ts, init_Tk=init_Tk).to(device, dtype)
    shared_fuse_expert = ShivaFuserExpert(
        hidden_dim=model_dim, ffn_mult=fuse_expert_ffn_mult
    ).to(device, dtype)
    grouped_routers = nn.ModuleList(
        [
            ShivaRouter(
                hidden_dim=model_dim,
                time_embed_dim=time_embed_dim,
                prompt_embed_dim=prompt_embed_dim,
                layer_embed_dim=layer_embed_dim,
                bottleneck_dim=bottleneck_dim,
            )
            for _ in range(num_router_groups)
        ]
    ).to(device, dtype)

    ## initialization: keep parameters to reasonable values
    nn.init.constant_(shared_gater_policy.net[-1].bias, 0.0)
    with torch.no_grad():
        shared_gater_policy.net[-1].bias[0] = 3.0   # Prune Logit
        shared_gater_policy.net[-1].bias[1] = -3.0  # Fuse Logit

    logit_bias = 2.5
    nn.init.constant_(shared_ratio_policy.net[-1].bias, logit_bias)
    nn.init.normal_(shared_ratio_policy.net[-1].weight, mean=0.0, std=1e-4)

    # target_r = 0.8
    # init_bias_val = torch.log(torch.tensor(target_r / (1 - target_r))).item()
    # nn.init.constant_(shared_ratio_policy.net[-1].bias, init_bias_val)
    # nn.init.normal_(shared_ratio_policy.net[-1].weight, mean=0.0, std=1e-4)

    for router in grouped_routers:
        nn.init.zeros_(router.router_head.weight)
        nn.init.constant_(router.router_head.bias, 1.0)

    nn.init.zeros_(shared_fuse_expert.fusion_ffn[-1].weight)
    nn.init.zeros_(shared_fuse_expert.fusion_ffn[-1].bias)
    nn.init.zeros_(shared_fuse_expert.gate_mlp[-2].weight)
    nn.init.constant_(shared_fuse_expert.gate_mlp[-2].bias, -5.0)

    layers_per_group = num_layers // num_router_groups

    injected = 0
    for i, (hook_name, module) in enumerate(all_blocks):
        router_group_id = min(i // layers_per_group, num_router_groups - 1)
        injected_router = grouped_routers[router_group_id]

        hook_kwargs = {
            "shared_layer_embedder": shared_layer_embedder,
            "shared_router": injected_router,
            "shared_gater_policy": shared_gater_policy,
            "shared_ratio_policy": shared_ratio_policy,
            "shared_pruner": shared_pruner_expert,
            "shared_context_pruner": shared_context_pruner,
            "shared_fuse_expert": shared_fuse_expert,
            **kwargs,
        }

        hook = hook_cls_to_install(**hook_kwargs)
        hook.register_name(model_type, hook_name)
        if hasattr(module, "register_hook"):
            module.register_hook(hook_name, hook)
        else:
            raise RuntimeError(
                f"Module '{hook_name}' does not support hook registration."
            )

        hook_manager_cls._hook_map[hook_name] = hook
        hook_manager_cls._weak_blocks[hook_name] = module
        injected += 1

    print(
        f"[ShivaInstaller] Injected {injected} shared hooks (using {num_router_groups} router groups)."
    )

if __name__ == "__main__":
    from shiva_dit.utils import get_pipeline, process_quick_start

    model_type = "sd3"
    pipe, model_attrs = get_pipeline(model_type)

    shiva_config = {
        "bottleneck_dim": 64,
        "layer_embed_dim": 64,
        "num_router_groups": 4,
        "fuse_expert_ffn_mult": 2,
        "context_pruning_threshold": None,
        "context_pruning_ratio": None,
        "use_gater_lut": False,
        "init_Ts": 5.0,
        "init_Tk": 5.0,
    }

    HookManager.configure_hooks(
        model_type=model_type,
        default=(ShivaDitBlockHook, shiva_config),
    )

    with HookManager.enabled():
        HookManager.inject_hooks(
            model_type=model_type, pipe=pipe, model_attrs=model_attrs
        )
        res = process_quick_start(pipe)
        res.save(f"{model_type}_shiva.png")

    print(global_context)
