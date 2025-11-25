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
    proj_dim = kwargs.pop("proj_dim", 128)
    
    # Common Configs
    init_Ts = kwargs.pop("init_Ts", 30.0)
    init_Tk = kwargs.pop("init_Tk", 30.0)

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
                total_layers=num_layers, ratio_bottleneck_dim=ratio_bottleneck_dim
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
    shared_context_pruner = None
    shared_context_ratio_policy = None

    if shiva_mode in ["context", "both"]:
        print("  > Instantiating Context Pruning Modules...")
        shared_context_pruner = ShivaContextPruner(
            img_dim=model_dim,
            txt_dim=prompt_embed_dim,
            proj_dim=proj_dim,
            init_Ts=init_Ts,
            init_Tk=init_Tk,
        ).to(device, torch.float32)

        if context_ratio_mode == "adaptive":
            shared_context_ratio_policy = ShivaRatioPolicy(
                total_layers=num_layers, ratio_bottleneck_dim=ratio_bottleneck_dim
            ).to(device, dtype)

    layers_per_group = num_layers // num_router_groups if num_router_groups > 0 else num_layers

    injected = 0
    for i, (hook_name, module) in enumerate(all_blocks):
        injected_router = None
        if grouped_routers is not None:
            router_group_id = min(i // layers_per_group, num_router_groups - 1)
            injected_router = grouped_routers[router_group_id]

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
            "shared_context_pruner": shared_context_pruner,
            "shared_context_ratio_policy": shared_context_ratio_policy,
            "fixed_context_ratio": fixed_context_ratio,
            
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

    print(f"[ShivaInstaller] Injected {injected} shared hooks.")

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
