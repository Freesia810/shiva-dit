import torch
import os
import glob
import json
from peft import LoraConfig as PeftLoraConfig, get_peft_model

from shiva_dit.utils import get_pipeline
from shiva_dit.hook import HookManager

from shiva_dit.shiva_hook import ShivaDitBlockHook
from baselines.tome_sd import ToMeSDDitBlockHook
from baselines.tofu import ToFuDitBlockHook
from baselines.ibtm import IBTMDiTBlockHook
from baselines.sdtm import SDTMDiTBlockHook
from baselines.toma import ToMADitBlockHook
from baselines.diffcr import DiffCRDitBlockHook
from baselines.dy_dit import DynamicDiTBlockHook
from baselines.sparse_dit import SparseDiTBlockHook

HOOK_CLASS_MAP = {
    "ToMeSDDitBlockHook": ToMeSDDitBlockHook,
    "ToFuDitBlockHook": ToFuDitBlockHook,
    "IBTMDiTBlockHook": IBTMDiTBlockHook,
    "SDTMDiTBlockHook": SDTMDiTBlockHook,
    "ToMADitBlockHook": ToMADitBlockHook,
    "DiffCRDitBlockHook": DiffCRDitBlockHook,
    "DynamicDiTBlockHook": DynamicDiTBlockHook,
    "SparseDiTBlockHook": SparseDiTBlockHook,
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
    meta_config: dict = None
):
    checkpoint_state = None
    load_format = "unknown"

    if ckpt_path is None:
        if meta_config is None:
            if model_type is None:
                raise ValueError("Baseline mode: 'model_type' required.")
            
            print(f"Loading Baseline Pipeline for: {model_type}")
            pipe, model_attrs = get_pipeline(model_type, model_path=base_model_path, dtype=dtype)
            pipe.to(device)
            return pipe
        else:
            if 'model_type' in meta_config:
                model_type = meta_config['model_type']
            else:
                raise ValueError("Meta Config mode: 'model_type' required.")
            print(f"Loading Meta Pipeline for: {model_type}")
            pipe, model_attrs = get_pipeline(model_type, model_path=base_model_path, dtype=dtype)
            pipe.to(device)
    else:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        if os.path.isdir(ckpt_path):
            config_path = os.path.join(ckpt_path, "shiva_config.json")
            
            if os.path.exists(config_path):
                print(f"Found metadata config at {config_path}")
                with open(config_path, "r") as f:
                    meta_config = json.load(f)
            
            modular_dir = os.path.join(ckpt_path, "modular_weights")
            if os.path.exists(modular_dir):
                load_format = "modular"
                print("Detected Format 2: Modular Weights Directory")
            else:
                potential_weights = glob.glob(os.path.join(ckpt_path, "*model*.safetensors")) + \
                                    glob.glob(os.path.join(ckpt_path, "*model*.bin"))
                if potential_weights:
                    load_format = "accelerator"
                    weights_path = potential_weights[0]
                    print(f"Detected Format 1: Accelerator State Directory (Found {os.path.basename(weights_path)})")
                    
                    if weights_path.endswith(".safetensors"):
                        from safetensors.torch import load_file as load_safetensors
                        checkpoint_state = load_safetensors(weights_path, device="cpu")
                    else:
                        checkpoint_state = torch.load(weights_path, map_location="cpu")
                else:
                    raise ValueError("Could not determine checkpoint format.")

        elif os.path.isfile(ckpt_path):
            load_format = "single_file"
            print("Detected Format 3: Single PT File")
            checkpoint_state = torch.load(ckpt_path, map_location="cpu")
            if isinstance(checkpoint_state, dict):
                if "model_type" in checkpoint_state: 
                    meta_config = {
                        "model_type": checkpoint_state.get("model_type"),
                        "training_mode": checkpoint_state.get("training_mode"),
                        "hook_info": checkpoint_state.get("hook_info"),
                        "lora_info": checkpoint_state.get("lora_info"),
                    }
                    checkpoint_state = checkpoint_state.get("model") # Extract weights
                else:
                    raise ValueError("Could not determine checkpoint format.")
            else:
                raise ValueError("Could not determine checkpoint format.")
        
    resolved_model_type = meta_config.get("model_type", model_type)
    if resolved_model_type is None:
        raise ValueError("Could not determine 'model_type' from checkpoint or arguments.")
    
    training_mode = meta_config.get("training_mode", "finetune") # Default assumption
    hook_info = meta_config.get("hook_info")
    lora_info = meta_config.get("lora_info")

    print(f"  - Model Type: {resolved_model_type}")
    print(f"  - Training Mode: {training_mode}")
    
    pipe, model_attrs = get_pipeline(
        resolved_model_type,
        model_path=base_model_path,
        dtype=dtype,
    )
    
    if hook_info:
        hook_class_name = hook_info.get("class_name")
        hook_kwargs = hook_info.get("config", {})
        print(f"  - Configuring Hook: {hook_class_name}")

        HookClass = HOOK_CLASS_MAP.get(hook_class_name)
        if HookClass:
            if hook_class_name == "ShivaDitBlockHook":
                hook_kwargs["capture_mask_for_viz"] = True
                
                # LUT logic (Keep existing logic)
                if gater_lut_path and os.path.exists(gater_lut_path):
                    lut_data = torch.load(gater_lut_path, map_location=device)
                    hook_kwargs["use_gater_lut"] = True
                    if isinstance(lut_data, dict):
                        hook_kwargs["shared_gater_cache"] = lut_data.get("gater")
                        hook_kwargs["shared_ratio_cache"] = lut_data.get("ratio")

                if optimize_hooks:
                    hook_kwargs["skip_layer_indices"] = [0]

            HookManager.configure_hooks(resolved_model_type, default=(HookClass, hook_kwargs))
            HookManager.enable()
            HookManager.inject_hooks(resolved_model_type, pipe=pipe, model_attrs=model_attrs)
    
    model = tuple(getattr(pipe, model) for model in model_attrs)[0]

    if training_mode == "lora" and lora_info:
        print("  - Applying LoRA adapter...")
        if "rank" in lora_info and "r" not in lora_info: lora_info["r"] = lora_info["rank"]
        peft_valid_keys = set(PeftLoraConfig.__annotations__.keys())
        filtered_lora_info = {k: v for k, v in lora_info.items() if k in peft_valid_keys or k == 'r'}
        peft_config = PeftLoraConfig(**filtered_lora_info, bias="none")
        model = get_peft_model(model, peft_config)

    print("  - Loading weights...")
    
    if load_format in ["single_file", "accelerator"] and checkpoint_state is not None:
        model.load_state_dict(checkpoint_state, strict=False)
    
    elif load_format in ["modular", "modular_flat"]:
        target_dir = modular_dir if load_format == "modular" else ckpt_path
        pt_files = glob.glob(os.path.join(target_dir, "*.pt"))
        print(f"    Found {len(pt_files)} modular files in {target_dir}")
        
        full_sd = {}
        for pt_f in pt_files:
            try:
                part_sd = torch.load(pt_f, map_location="cpu")
                full_sd.update(part_sd)
            except Exception as e:
                print(f"    Error loading {pt_f}: {e}")
        
        model.load_state_dict(full_sd, strict=False)

    if training_mode == "lora" and merge_lora:
        print("  - Merging LoRA weights...")
        try:
            model = model.merge_and_unload()
            setattr(pipe, model_attrs[0], model)
        except Exception as e:
            print(f"    Merge failed (maybe not supported for this PEFT version?): {e}")

    pipe.to(device)
    model.eval()
    
    for hook in HookManager.get_all_hooks().values():
        if isinstance(hook, torch.nn.Module):
            hook.eval().requires_grad_(False)
            
    print("Pipeline ready.")
    return pipe