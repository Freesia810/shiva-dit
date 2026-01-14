# Shiva-DiT Hook Mechanism

The `shiva_dit.hook` module provides a flexible, non-intrusive way to inject custom logic (such as token pruning, merging, or logging) into standard Diffusion Transformer (DiT) blocks. It abstracts the differences between various architectures (SD3, Flux, PixArt, etc.), allowing you to write model-agnostic acceleration code.

## Core Architecture

The system operates using a **HookManager** that injects a **BaseDiTBlockHook** into the target model's modules. The interaction between the hook and the original model block is controlled by two distinct modes: **Callback** and **Delegate**.

### 1. Callback Mode (Standard Interception)
In this mode, the original `forward` method of the block is preserved. The hook inserts specific "checkpoints" (callbacks) at key stages of the execution flow.

* **Best for:** Lightweight modifications (e.g., logging, feature extraction) or standard token merging/pruning that fits into the pre-defined flow.
* **Mechanism:** The hook overrides specific methods like `hook_pre_attn` or `hook_after_module`.

**Hook Points (Forward Pass):**
1.  **`hook_pre_norm`**: Before LayerNorm/Pre-processing, referred as the entry of the block normally.
2.  **`hook_after_norm`**: After LayerNorm/Pre-processing, before the module (Attn/FFN).
3.  **`hook_after_module`**: After the block's main computation (before residual connection).
4.  **`hook_pre_attn`**(experimental): Before attention calculation. Ideal for manipulating Q/K/V.
5.  **`hook_after_attn`**(experimental): After the attention matrix is computed. Useful for visualizing attention maps.

The `hook_pre_norm`, `hook_after_norm` and `hook_after_module` methods all utilize the module_name argument to identify the target component (e.g., "self-attn", "cross-attn", or "ffn"). Notably, for Flux-1.dev's SingleDiT blocks, the "parallel" module name is used.

**Example:**
```python
class MyLoggerHook(BaseDiTBlockHook):
    def hook_pre_norm(
        self, 
        module_name: Literal["self-attn", "cross-attn", "ffn", "parallel"],
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        temb: torch.Tensor = None,
    ):
        if image_rotary_emb is not None:
            return hidden_states, encoder_hidden_states, image_rotary_emb
        return hidden_states, encoder_hidden_states
```

### 2. Delegate Mode (Full Control)
In this mode, the hook **completely takes over** the execution of the block's `forward` method. This is achieved via the `@delegated_forward` decorator.

* **Best for:** Best for: Complex architectural changes (e.g., changing the order of operations, replacing Attention/FFN entirely) or when the standard callback points are insufficient.

* **Mechanism:** The `HookManager` replaces the block's `forward` method with a wrapper. If the hook implements `delegated_call`, it receives the original function and all arguments, allowing it to decide *if, when, and how* to call the original logic.

**Implementation Guide**: To use this, your hook class must inherit from `BaseDelegator` and override `delegated_call`.

```python
class MyCustomControlHook(BaseDiTBlockHook):
    def delegated_call(self, original_forward_func, module_self, *args, **kwargs):
        # 1. Pre-processing logic (e.g., manipulate inputs)
        # inspect module is more recommended to bind kwargs and default args automatically
        sig = inspect.signature(func)
        bound = sig.bind_partial(module_self, *args, **kwargs)
        bound.apply_defaults()
        hidden_states = bound.arguments.get("hidden_states")
        if self.should_skip_layer(hidden_states):
            return hidden_states  # Skip original computation entirely
        
        # 2. Call original forward (or parts of it)
        output = original_forward_func(module_self, *args, **kwargs)
        
        # 3. Post-processing logic
        return output
```

In practice, these two modes can be used in conjunction; all hook methods will continue to be triggered as expected.

## HookManager
The `HookManager` is a static utility class responsible for configuration, injection, and lifecycle management.

**Key Features**

* **Model Agnostic:** Automatically resolves block types for SD3, Flux, PixArt, Qwen-Image, and Wan-Video.

* **Context Management:** Use with `HookManager.enabled():` to ensure hooks are cleanly removed after inference.

* **Shared State:** Supports a global buffer for passing data between layers (e.g., layer-wise communication).

### Usage Guide
**Step 1: Define Your Hook**

```python
from shiva_dit.hook.block_hook import BaseDiTBlockHook

class MyHook(BaseDiTBlockHook):
    # Implement callback methods or delegated_call
    pass
```

**Step 2: Configure & Inject**
```python
from shiva_dit.hook.hook_manager import HookManager

# Configuration: Map your hook class to the model
hook_config = {"param_a": 10} 
HookManager.configure_hooks(
    model_type="sd3",
    default=(MyHook, hook_config)
)

# Execution
pipe = ... # Load your pipeline
with HookManager.enabled():
    # Inject into specific components (e.g., "transformer")
    HookManager.inject_hooks(
        model_type="sd3", 
        pipe=pipe, 
        model_attrs=["transformer"]
    )
    
    # Run Inference
    output = pipe("A photo of a cat")
```

**Advanced: Custom Layer Targeting**
By default, hooks are applied to all Transformer blocks. You can target specific layers using Unix shell-style wildcards in `configure_hooks`:

```python
HookManager.configure_hooks(
    model_type="sd3",
    custom={
        "transformer_blocks.0": (FirstLayerHook, {}),   # Only Layer 0
        "transformer_blocks.*": (RestLayerHook, {}),    # All other layers
    }
)
```