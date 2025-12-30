import torch.nn as nn
from typing import Type, Optional, List

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
    skip_layer_indices: Optional[List[int]] = None,
    **kwargs,
):
    shiva_mode = kwargs.pop("shiva_mode", "image")  # "image", "context", "both"

    # Image Configs
    image_ratio_mode = kwargs.pop("image_ratio_mode", "adaptive")
    fixed_image_ratio = kwargs.pop("fixed_image_ratio", 0.5)
    bottleneck_dim = kwargs.pop("bottleneck_dim", 64)
    ratio_bottleneck_dim = kwargs.pop("ratio_bottleneck_dim", 32)
    num_router_groups = kwargs.pop("num_router_groups", 4)
    layer_embed_dim = kwargs.pop("layer_embed_dim", 256)
    fuse_expert_ffn_mult = kwargs.pop("fuse_expert_ffn_mult", 2)

    # Context Configs
    context_ratio_mode = kwargs.pop("context_ratio_mode", "fixed")
    fixed_context_ratio = kwargs.pop("fixed_context_ratio", 1.0)
    context_pruner_mode = kwargs.pop("context_pruner_mode", "learned")
    context_pool_size = kwargs.pop("context_pool_size", None)
    context_proj_dim = kwargs.pop("context_proj_dim", 128)
    context_shared_config = kwargs.pop("context_shared_config", {})

    # Common Configs
    init_Ts = kwargs.pop("init_Ts", 30.0)
    init_Tk = kwargs.pop("init_Tk", 30.0)

    shared_gater_cache = kwargs.pop("shared_gater_cache", None)
    shared_ratio_cache = kwargs.pop("shared_ratio_cache", None)
    shared_context_ratio_cache = kwargs.pop("shared_context_ratio_cache", None)

    model_dim = global_context.hidden_dim
    prompt_embed_dim = global_context.encoder_hidden_dim
    if model_dim is None:
        raise ValueError("[ShivaInstaller] 'hidden_dim' not found in global_context.")
    if prompt_embed_dim is None:
        raise ValueError("[ShivaInstaller] 'encoder_hidden_dim' not found in global_context.")

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

    print(f"[ShivaInstaller] Found {num_layers} blocks. Mode: {shiva_mode}")

    # --- 1. Image Pruning Modules ---
    router_layer_embedder = None
    shared_gater_policy = None
    shared_image_ratio_policy = None
    shared_pruner_expert = None
    shared_fuse_expert = None
    grouped_routers = None

    if shiva_mode in ["image", "both"]:
        print("  > Instantiating Image Pruning Modules...")

        router_layer_embedder = nn.Embedding(num_layers, layer_embed_dim).to(device, dtype)

        shared_gater_policy = ShivaGaterPolicy(
            time_embed_dim=time_embed_dim, 
            layer_embed_dim=layer_embed_dim, 
            bottleneck_dim=bottleneck_dim,
            num_layers=num_layers
        ).to(device, dtype)

        if image_ratio_mode == "adaptive":
            shared_image_ratio_policy = ShivaRatioPolicy(
                total_layers=num_layers, ratio_bottleneck_dim=ratio_bottleneck_dim, init_ratio=fixed_image_ratio, time_embed_dim=time_embed_dim
            ).to(device, dtype)

        shared_pruner_expert = ShivaPrunerExpert(init_Ts=init_Ts, init_Tk=init_Tk).to(device, dtype)
        shared_fuse_expert = ShivaFuserExpert(
            hidden_dim=model_dim, ffn_mult=fuse_expert_ffn_mult
        ).to(device, dtype)

        grouped_routers = nn.ModuleList([
            ShivaRouter(
                hidden_dim=model_dim,
                time_embed_dim=time_embed_dim,
                prompt_embed_dim=prompt_embed_dim,
                layer_embed_dim=layer_embed_dim,
                bottleneck_dim=bottleneck_dim,
            )
            for _ in range(num_router_groups)
        ]).to(device, dtype)

    # --- 2. Context Pruning Modules ---
    shared_context_ratio_policy = None

    if shiva_mode in ["context", "both"]:
        print("  > Instantiating Context Modules...")

        if context_ratio_mode == "adaptive":
            shared_context_ratio_policy = ShivaRatioPolicy(
                total_layers=num_layers, ratio_bottleneck_dim=ratio_bottleneck_dim, init_ratio=fixed_context_ratio
            ).to(device, dtype)
            assert False, "Adaptive context ratio is not implemented yet."
        elif context_ratio_mode == 'linear':
            assert context_pruner_mode == 'free', "Linear context ratio scheduling only supports 'free' pruner mode."

            spatial_step_thresh = spatial_group_size = None
            temporal_layer_thresh = temporal_interval = None
            if (
                "spatial_step_thresh" in context_shared_config
                or "spatial_group_size" in context_shared_config
            ):
                spatial_step_thresh = context_shared_config["spatial_step_thresh"]
                spatial_group_size = context_shared_config["spatial_group_size"]
                assert (
                    0 < spatial_step_thresh <= 1.0
                ), "'spatial_step_thresh' must be in (0, 1]."

            if (
                "temporal_layer_thresh" in context_shared_config
                or "temporal_interval" in context_shared_config
            ):
                temporal_layer_thresh = context_shared_config["temporal_layer_thresh"]
                temporal_interval = context_shared_config["temporal_interval"]
                if temporal_layer_thresh <= 1.0:
                    temporal_layer_thresh = int(temporal_layer_thresh * num_layers)
                assert (
                    0 < temporal_layer_thresh <= num_layers
                ), f"'temporal_layer_thresh' must be in (0, {num_layers}]."

            if spatial_step_thresh and spatial_group_size:
                print(f"    - Spatial Linear Scheduling: {spatial_step_thresh} steps with group size {spatial_group_size}")
                HookManager.set_global(
                    "shiva_context_shared_spatial",
                    {
                        "spatial_step_thresh": spatial_step_thresh,
                        "spatial_group_size": spatial_group_size,
                    },
                )
            if temporal_layer_thresh and temporal_interval:
                print(f"    - Temporal Linear Scheduling: {temporal_layer_thresh} layers with interval {temporal_interval}")
                HookManager.set_global(
                    "shiva_context_shared_temporal",
                    {
                        "temporal_layer_thresh": temporal_layer_thresh,
                        "temporal_interval": temporal_interval,
                    },
                )

            print(f"  > Instantiating Linear Schedule Context Pruning Modules... {fixed_context_ratio[0]} -> {fixed_context_ratio[1]}")

    layers_per_group = num_layers // num_router_groups if num_router_groups > 0 else num_layers

    if skip_layer_indices is None:
        skip_layer_indices = []

    injected = 0
    skipped = 0

    for i, (hook_name, module) in enumerate(all_blocks):
        if i in skip_layer_indices:
            skipped += 1
            continue

        # Image Router
        injected_router = None
        if grouped_routers is not None:
            router_group_id = min(i // layers_per_group, num_router_groups - 1)
            injected_router = grouped_routers[router_group_id]

        # Context Router, per layer
        injected_context_router = None
        if shiva_mode in ["context", "both"]:
            injected_context_router = ShivaContextPruner(
                mode=context_pruner_mode,
                pool_size=context_pool_size,
                img_dim=model_dim,
                txt_dim=prompt_embed_dim,
                proj_dim=context_proj_dim,
                init_Ts=init_Ts,
                init_Tk=init_Tk,
            ).to(device, dtype)

        hook_kwargs = {
            "shiva_mode": shiva_mode,
            "total_layers": num_layers,
            # Image Modules & Embeddings
            "router_layer_embedder": router_layer_embedder,
            "shared_router": injected_router,
            "shared_gater_policy": shared_gater_policy,
            "shared_image_ratio_policy": shared_image_ratio_policy,
            "shared_pruner": shared_pruner_expert,
            "shared_fuse_expert": shared_fuse_expert,
            "fixed_image_ratio": fixed_image_ratio,
            # Context Modules
            "shared_context_pruner": injected_context_router,
            "shared_context_ratio_policy": shared_context_ratio_policy,
            "fixed_context_ratio": fixed_context_ratio,
            # LuT
            "shared_gater_cache": shared_gater_cache,
            "shared_ratio_cache": shared_ratio_cache,
            "shared_context_ratio_cache": shared_context_ratio_cache,
            "total_layers": num_layers,
            **kwargs,
        }

        hook = hook_cls_to_install(**hook_kwargs)
        hook.register_name(model_type, hook_name)
        if hasattr(module, "register_hook"):
            module.register_hook(hook_name, hook)
        else:
            raise RuntimeError(f"Module '{hook_name}' does not support hook registration.")

        hook_manager_cls._hook_map[hook_name] = hook
        hook_manager_cls._weak_blocks[hook_name] = module
        injected += 1

        if context_pruner_mode == "free":
            hook.eval()

    print(f"[ShivaInstaller] Injected {injected} shared hooks.")

if __name__ == "__main__":
    from shiva_dit.utils import get_pipeline, process_quick_start
    from PIL import Image

    model_type = "qwen-image-edit"
    pipe, model_attrs = get_pipeline(model_type)
    prompt = "Change the rabbit's color to purple, with a flash light background."
    image = Image.open("rabbitfeatured.jpg").convert("RGB")

    shiva_config = {
        "shiva_mode": "context",
        "context_ratio_mode": "linear",
        "fixed_context_ratio": (1.0, 0.3),
        "context_pruner_mode": "free",
        "context_pool_size": None,
        "context_shared_config": {
            "spatial_step_thresh": 1.0,
            "spatial_group_size": 60,
            "temporal_layer_thresh": 60,
            "temporal_interval": 50,
        },
        "init_Ts": 10.0,
        "init_Tk": 10.0,
    }

    HookManager.configure_hooks(
        model_type=model_type,
        default=(ShivaDitBlockHook, shiva_config),
    )

    with HookManager.enabled():
        HookManager.inject_hooks(
            model_type=model_type, pipe=pipe, model_attrs=model_attrs
        )
        res = process_quick_start(pipe, prompt=prompt, max_sequence_length=1024, image=image)
        res.save(f"{model_type}_shiva.png")

    print(global_context)
