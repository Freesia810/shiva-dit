import torch.nn as nn
import weakref
import fnmatch
from typing import Dict, Type, Optional, Union
from contextlib import contextmanager
from copy import deepcopy


# SD3 -> JointTransformerBlock
# FLUX-dev/FLUX-kontext -> FluxTransformerBlock & FluxSingleTransformerBlock
# qwen-image/qwen-image-edit -> QwenImageTransformerBlock
# pixart-sigma -> BasicTransformerBlock
# wan -> WanTransformerBlock


class HookManager:
    _enabled: bool = False
    _hook_config: Dict[str, tuple[Type["BaseDiTBlockHook"], dict]] = {}
    _hook_map: Dict[Union[int, str], "BaseDiTBlockHook"] = {}
    _weak_blocks: "weakref.WeakValueDictionary[str, nn.Module]" = weakref.WeakValueDictionary()

    _block_name = {
        'sd3': ('transformer.transformer_blocks.*',),
        'flux-dev': ('transformer.transformer_blocks.*', 'transformer.single_transformer_blocks.*'),
        'qwen-image': ('transformer.transformer_blocks.*',),
        'qwen-image-edit': ('transformer.transformer_blocks.*',),
        'pixart-sigma': ('transformer.transformer_blocks.*',),
        'wan': ('transformer.blocks.*', 'transformer_2.blocks.*'),
    }

    @classmethod
    def enable(cls):
        cls._enabled = True
    
    @classmethod
    def disable(cls):
        cls._enabled = False
        for name, hook in list(cls._hook_map.items()):
            if hasattr(hook, "close"):
                try:
                    hook.close()
                except Exception as e:
                    print(f"[HookManager] Warning: failed to close hook {name}: {e}")
        cls._hook_map.clear()
        cls._weak_blocks.clear()
        cls._hook_config.clear()

    @classmethod
    @contextmanager
    def enabled(cls):
        cls.enable()
        try:
            yield
        finally:
            cls.disable()
    
    @classmethod
    def configure_hooks(
        cls,
        model_type: str,
        default: Optional[tuple[Type["BaseDiTBlockHook"], dict]] = None,
        custom: Optional[Dict[str, tuple[Type["BaseDiTBlockHook"], dict]]] = None,
    ):
        cls._hook_config.clear()

        if default:
            hook_cls, kwargs = default
            if not isinstance(kwargs, dict):
                raise TypeError("default hook kwargs must be a dict")
            cls._hook_config["__default__"] = (hook_cls, kwargs)

        if custom:
            valid_patterns = cls._block_name.get(model_type, None)
            if not valid_patterns:
                raise ValueError(f"Unknown model_type '{model_type}'")

            for pattern, (hook_cls, kwargs) in custom.items():
                if not any(fnmatch.fnmatchcase(pattern, valid) for valid in valid_patterns):
                    raise ValueError(f"Unknown layer pattern '{pattern}' for model '{model_type}'")
                if not isinstance(kwargs, dict):
                    raise TypeError(f"Hook config for '{pattern}' must be a dict")
                cls._hook_config[pattern] = (hook_cls, kwargs)
    
    @classmethod
    def inject_hooks(cls, model_type: str, model: nn.Module):
        if not cls._enabled:
            print("[HookManager] Disabled, skipping hook injection.")
            return

        patterns = cls._block_name.get(model_type, None)
        if not patterns:
            raise ValueError(f"Unknown model_type '{model_type}'")

        injected = 0

        for name, module in model.named_modules():
            if any(fnmatch.fnmatchcase(name, p) for p in patterns):
                hook_cls, kwargs = cls._resolve_hook_config(name)
                hook = hook_cls(**deepcopy(kwargs))
                module._hook = hook
                cls._hook_map[name] = hook
                cls._weak_blocks[name] = module
                injected += 1
                print(f"[HookManager] Injected hook for layer '{name}', hook class: {hook_cls}")

        print(f"[HookManager] Injected {injected} hooks for model '{model_type}'.")

    @classmethod
    def _resolve_hook_config(cls, name: str):
        """Decide whether to use custom or default hook based on layer name."""
        for pattern, (hook_cls, kwargs) in cls._hook_config.items():
            if pattern != "__default__" and fnmatch.fnmatchcase(name, pattern):
                return hook_cls, kwargs
        if "__default__" in cls._hook_config:
            return cls._hook_config["__default__"]
        raise RuntimeError(f"No hook config found for '{name}' and no default hook is set.")
    
    @classmethod
    def info(cls):
        print(f"[HookManager] enabled={cls._enabled}, hooks={len(cls._hook_map)}")
        for name, hook in cls._hook_map.items():
            print(f"  {name}: {hook.__class__.__name__}")