import torch
import os
from peft import LoraConfig as PeftLoraConfig, get_peft_model
from dataclasses import asdict

from project_name.utils import get_pipeline
from project_name.hook import HookManager

from baselines.diffcr import DiffCRDitBlockHook
from baselines.dy_dit import DynamicDiTBlockHook

HOOK_CLASS_MAP = {
    "DiffCRDitBlockHook": DiffCRDitBlockHook,
    "DynamicDiTBlockHook": DynamicDiTBlockHook,
}

def load_pipeline_for_inference(
    ckpt_path: str, 
    base_model_path: str = None, 
    device="cuda", 
    dtype=torch.bfloat16
):    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading inference checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    model_type = checkpoint.get("model_type")
    training_mode = checkpoint.get("training_mode")
    hook_info = checkpoint.get("hook_info")
    lora_info = checkpoint.get("lora_info")
    model_state = checkpoint.get("model")

    if not model_type:
        raise ValueError("Checkpoint is invalid: missing 'model_type' metadata.")
    
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
        print(f"  - Applying Hook: {hook_class_name}")

        HookClass = HOOK_CLASS_MAP.get(hook_class_name)
        if not HookClass:
            raise ValueError(f"Unknown hook class '{hook_class_name}'. Please register it in HOOK_CLASS_MAP in inference_utils.py")

        HookManager.configure_hooks(
            model_type=model_type,
            default=(HookClass, hook_kwargs),
        )
        HookManager.enable()
        HookManager.inject_hooks(
            model_type=model_type,
            pipe=pipe,
            model_attrs=model_attrs,
        )

    model = tuple(
        getattr(pipe, model) for model in model_attrs
    )[0]

    if lora_info and training_mode == "lora":
        print("  - Applying LoRA adapter...")
        
        peft_config_keys = PeftLoraConfig.__annotations__.keys()
        filtered_lora_info = {k: v for k, v in lora_info.items() if k in peft_config_keys}
        
        peft_config = PeftLoraConfig(
            **filtered_lora_info,
            bias="none"
        )
        
        model = get_peft_model(model, peft_config)
        print("  - LoRA applied.")

    print("  - Loading saved weights into model...")
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    
    if unexpected:
        print(f"  - Warning: Unexpected keys in checkpoint: {unexpected}")
    
    print(f"  - Info: Ignored {len(missing)} missing keys (base model params).")
    print("  - Weights loaded successfully.")

    setattr(pipe, model_attrs[0], model)
    
    pipe.to(device)
    model.eval()
    
    for hook in HookManager.get_all_hooks().values():
        if isinstance(hook, torch.nn.Module):
            hook.eval()
    
    print(f"\nPipeline ready for inference on device '{device}'.")
    return pipe