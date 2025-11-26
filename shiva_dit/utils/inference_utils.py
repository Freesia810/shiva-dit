import torch
import os
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
    ckpt_path: str = None,
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
        return pipe
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading inference checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    ckpt_model_type = checkpoint.get("model_type")
    if model_type is not None and ckpt_model_type != model_type:
        print(f"Warning: CLI model_type '{model_type}' differs from Checkpoint '{ckpt_model_type}'. Using Checkpoint's type.")
    
    model_type = ckpt_model_type
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

        hook_kwargs["capture_mask_for_viz"] = False

        if gater_lut_path:
            if os.path.exists(gater_lut_path):
                print(f"  - Loading LUT from {gater_lut_path}")
                lut_data = torch.load(gater_lut_path, map_location="cpu")
                
                hook_kwargs["use_gater_lut"] = True
                
                if isinstance(lut_data, dict):
                    hook_kwargs["shared_gater_cache"] = lut_data.get("gater")
                    hook_kwargs["shared_ratio_cache"] = lut_data.get("ratio")
                    print("    > Loaded both Gater and Ratio tables.")
                else:
                    print("    > Warning: Unknown LUT format. Ignoring.")
                    hook_kwargs["use_gater_lut"] = False
            else:
                print(f"  - Warning: LUT path {gater_lut_path} provided but file not found.")

        skip_layer_indices = []
        if optimize_hooks:
            print("  - Optimization enabled: Skipping Layer 0 injection.")
            skip_layer_indices.append(0)

        HookManager.configure_hooks(
            model_type=model_type,
            default=(HookClass, hook_kwargs),
        )
        HookManager.enable()
        
        print("  - Injecting Hooks into pipeline...")
        install_shiva_hooks(
            HookManager,
            model_type,
            pipe,
            model_attrs,
            HookClass,
            skip_layer_indices=skip_layer_indices, 
            **hook_kwargs
        )

    model = tuple(getattr(pipe, model) for model in model_attrs)[0]

    if lora_info and training_mode == "lora":
        print("  - Applying LoRA adapter...")
        
        if "rank" in lora_info and "r" not in lora_info:
            lora_info["r"] = lora_info["rank"]

        peft_valid_keys = set(PeftLoraConfig.__annotations__.keys())
        filtered_lora_info = {k: v for k, v in lora_info.items() if k in peft_valid_keys or k == 'r'}
        
        peft_config = PeftLoraConfig(**filtered_lora_info, bias="none")
        model = get_peft_model(model, peft_config)
        
        print("  - Loading LoRA weights from checkpoint...")
        model.load_state_dict(model_state, strict=False)
        
        if merge_lora:
            print("  - Merging LoRA weights into base model for faster inference...")
            model = model.merge_and_unload()
            setattr(pipe, model_attrs[0], model)
            print("  - Merge complete.")
        else:
            print("  - Keeping LoRA weights separate.")
            
    elif training_mode in ["finetune", "hook_only"]:
        print("  - Loading full model weights...")
        model.load_state_dict(model_state, strict=False)

    pipe.to(device)
    model.eval()
    
    for hook in HookManager.get_all_hooks().values():
        if isinstance(hook, torch.nn.Module):
            hook.eval()
            hook.requires_grad_(False)
    
    print(f"\nPipeline ready for inference on device '{device}'.")
    return pipe