import torch.nn as nn
import weakref
import fnmatch
from typing import Dict, Type, Optional, Union, Any
from contextlib import contextmanager
from copy import deepcopy
from shiva_dit.hook.block_hook import BaseDiTBlockHook


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
    _shared_buffer: Dict[str, Any] = {}

    _block_name = {
        'sd3': ('transformer_blocks.*',),
        'flux-dev': ('transformer_blocks.*', 'single_transformer_blocks.*'),
        'flux-kontext': ('transformer_blocks.*', 'single_transformer_blocks.*'),
        'qwen-image': ('transformer_blocks.*',),
        'qwen-image-edit': ('transformer_blocks.*',),
        'pixart-sigma': ('transformer_blocks.*',),
        'wan': ('blocks.*',),
    }

    @classmethod
    def _resolve_block_type(cls, model_type: str):
        """
        Lazy import transformer block types to avoid circular imports.
        """
        if model_type in ('sd3',):
            from diffusers.models.attention import JointTransformerBlock
            return JointTransformerBlock

        elif model_type in ('pixart-sigma',):
            from diffusers.models.attention import BasicTransformerBlock
            return BasicTransformerBlock

        elif model_type in ('flux-dev', 'flux-kontext'):
            from diffusers.models.transformers.transformer_flux import FluxTransformerBlock, FluxSingleTransformerBlock
            return (FluxTransformerBlock, FluxSingleTransformerBlock)

        elif model_type in ('qwen-image', 'qwen-image-edit'):
            from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock
            return QwenImageTransformerBlock

        elif model_type == 'wan':
            from diffusers.models.transformers.transformer_wan import WanTransformerBlock
            return WanTransformerBlock

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    @classmethod
    def get_block_type(cls, model_type: str):
        """Public accessor — cached resolver."""
        if not hasattr(cls, "_block_type_cache"):
            cls._block_type_cache = {}
        if model_type not in cls._block_type_cache:
            cls._block_type_cache[model_type] = cls._resolve_block_type(model_type)
        return cls._block_type_cache[model_type]

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
        cls._shared_buffer.clear()

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
            cls._hook_config["__default__"] = (hook_cls, deepcopy(kwargs))

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
    def inject_hooks(cls, model_type: str, pipe, model_attrs: list[str]):
        if not cls._enabled:
            print("[HookManager] Disabled, skipping hook injection.")
            return

        if "__default__" not in cls._hook_config:
            raise RuntimeError(
                "No default hook configured. Please call configure_hooks() first."
            )

        default_hook_cls, default_kwargs = cls._hook_config["__default__"]

        if default_hook_cls.__name__ == "ShivaDitBlockHook":
            print(f"[HookManager] Detected {default_hook_cls.__name__}.")
            print("[HookManager] Delegating injection logic to ShivaInstaller...")

            assert (
                len(cls._hook_config) == 1
            ), "[HookManager] When using shared hook injection (Shiva), only the default hook config is supported."

            try:
                from shiva_dit.shiva_installer import install_shiva_hooks
            except ImportError:
                print(
                    "Error: Could not import install_shiva_hooks. Make sure shiva_dit/shiva_installer.py exists."
                )
                raise

            install_shiva_hooks(
                cls,
                model_type,
                pipe,
                model_attrs,
                default_hook_cls,
                **default_kwargs,
            )

            print("[HookManager] Shared hook injection complete.")
        else:
            print(
                f"[HookManager] Using standard per-block injection logic (e.g., DiffCR)."
            )
            injected = 0

            for model_attr in model_attrs:
                m = getattr(pipe, model_attr)
                if not isinstance(m, nn.Module):
                    raise TypeError(f"Expected nn.Module, got {type(m)}")

                for name, module in m.named_modules():
                    if isinstance(module, cls.get_block_type(model_type)):
                        hook_cls, kwargs = cls._resolve_hook_config(name)
                        hook = hook_cls(**deepcopy(kwargs))  # 关键: deepcopy
                        hook_name = f"{model_attr}.{name}"

                        hook.register_name(model_type, hook_name)
                        if hasattr(module, "register_hook"):
                            module.register_hook(hook_name, hook)
                        else:
                            raise RuntimeError(
                                f"Module '{hook_name}' does not support hook registration."
                            )

                        cls._hook_map[hook_name] = hook
                        cls._weak_blocks[hook_name] = module
                        injected += 1
                        print(
                            f"[HookManager] Injected hook for layer '{hook_name}', hook class: {hook_cls.__name__}"
                        )

            print(f"[HookManager] Injected {injected} independent hooks.")

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
        print(f"[HookManager] Shared Buffer Keys: {list(cls._shared_buffer.keys())}")

    @classmethod
    def get_all_hooks(cls) -> Dict[Union[int, str], "BaseDiTBlockHook"]:
        return cls._hook_map

    @classmethod
    def set_global(cls, key: str, value: Any):
        if not cls._enabled:
            print("[HookManager] Warning: Disabled, skipping set_global.")
            return
        cls._shared_buffer[key] = value

    @classmethod
    def get_global(cls, key: str, default: Any = None) -> Any:
        return cls._shared_buffer.get(key, default)

    @classmethod
    def global_buffer(cls) -> Dict[str, Any]:
        return cls._shared_buffer
