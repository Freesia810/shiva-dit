import torch
import os
import json
from peft import LoraConfig as PeftLoraConfig, get_peft_model
from dataclasses import asdict

from shiva_dit.utils import get_pipeline
from shiva_dit.hook import HookManager
from shiva_dit.shiva_installer import install_shiva_hooks

from shiva_dit.shiva_hook import ShivaDitBlockHook
from baselines.diffcr import DiffCRDitBlockHook
from baselines.dy_dit import DynamicDiTBlockHook

HOOK_CLASS_MAP = {
    "DiffCRDitBlockHook": DiffCRDitBlockHook,
    "DynamicDiTBlockHook": DynamicDiTBlockHook,
    "ShivaDitBlockHook": ShivaDitBlockHook,
}

def load_pipeline_for_inference(
    ckpt_path: str, 
    base_model_path: str = None, 
    model_type: str = None,
    device="cuda", 
    dtype=torch.bfloat16,
    gater_lut_path: str = None,
    optimize_hooks: bool = True,
    merge_lora: bool = True,
):    
    if ckpt_path is None:
        if model_type is None:
            raise ValueError("Using Baseline mode (no ckpt_path): 'model_type' must be provided (e.g., 'flux-dev', 'sd3').")
        
        print(f"Loading Baseline Pipeline (No Checkpoint) for type: {model_type}")
        pipe, model_attrs = get_pipeline(
            model_type,
            model_path=base_model_path,
            dtype=dtype,
        )
        pipe.to(device)
        
        for attr in model_attrs:
            model_ref = getattr(pipe, attr)
            model_ref.eval()
            model_ref.requires_grad_(False)
            
        print(f"Baseline pipeline ready on '{device}'.")
        return 
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading inference checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    ckpt_model_type = checkpoint.get("model_type")
    if model_type is not None and ckpt_model_type != model_type:
        print(f"Warning: model_type '{model_type}' differs from Checkpoint '{ckpt_model_type}'. Using Checkpoint's type.")

    model_type = checkpoint.get("model_type")
    training_mode = checkpoint.get("training_mode")
    hook_info = checkpoint.get("hook_info")
    lora_info = checkpoint.get("lora_info")
    model_state = checkpoint.get("model")

    print(f"  - Model Type: {model_type}")
    print(f"  - Training Mode: {training_mode}")

    pipe, model_attrs = get_pipeline(
        model_type,
        model_path=base_model_path,
        dtype=dtype,
    )
    
    if hook_info:
        hook_class_name = hook_info.get("class_name")
        hook_kwargs = hook_info.get("config", {})
        print(f"  - Configuring Hook: {hook_class_name}")

        HookClass = HOOK_CLASS_MAP.get(hook_class_name)
        if not HookClass:
            raise ValueError(f"Unknown hook class '{hook_class_name}'.")

        if gater_lut_path:
            if os.path.exists(gater_lut_path):
                print(f"  - Loading Gater LUT from {gater_lut_path}")
                gater_cache = torch.load(gater_lut_path, map_location=device)
                hook_kwargs["use_gater_lut"] = True
                hook_kwargs["shared_gater_cache"] = gater_cache
            else:
                print(f"  - Warning: LUT path {gater_lut_path} provided but file not found. Ignoring.")
        
        skip_layer_indices = []
        if optimize_hooks:
            print("  - Optimization enabled: Skipping Layer 0 injection.")
            skip_layer_indices.append(0)

        HookManager.configure_hooks(
            model_type=model_type,
            default=(HookClass, hook_kwargs),
        )
        HookManager.enable()
        
        print("  - Injecting Hooks...")
        install_shiva_hooks(
            HookManager,
            model_type,
            pipe,
            model_attrs,
            HookClass,
            skip_layer_indices=skip_layer_indices,
            **hook_kwargs
        )

    model = tuple(
        getattr(pipe, model) for model in model_attrs
    )[0]

    if lora_info and training_mode == "lora":
        print("  - Applying LoRA adapter...")
        
        peft_config_keys = PeftLoraConfig.__annotations__.keys()
        filtered_lora_info = {k: v for k, v in lora_info.items() if k in peft_config_keys}
        
        peft_config = PeftLoraConfig(**filtered_lora_info, bias="none")
        model = get_peft_model(model, peft_config)
        
        print("  - Loading LoRA weights from checkpoint...")
        missing, unexpected = model.load_state_dict(model_state, strict=False)

        if merge_lora:
            print("  - Merging LoRA weights into base model for faster inference...")
            model = model.merge_and_unload()
            setattr(pipe, model_attrs[0], model)
    
    elif training_mode == "finetune" or training_mode == "hook_only":
        print("  - Loading model weights...")
        model.load_state_dict(model_state, strict=False)

    pipe.to(device)
    model.eval()
    
    for hook in HookManager.get_all_hooks().values():
        if isinstance(hook, torch.nn.Module):
            hook.eval()
            hook.requires_grad_(False)
    
    print(f"\nPipeline ready for inference on device '{device}'.")
    return pipe