# DiT block hooks

[HookManager](hook_manager.py) attaches a hook to compatible transformer blocks in the bundled Diffusers implementation. [Shiva's installer](../shiva_installer.py) creates the shared router, ratio policy, and pruning modules and connects them to the blocks.

Hooks can intercept individual block operations or control the full block forward pass.

| Mode | Interface | Role |
| --- | --- | --- |
| Callback | [BaseDiTBlockHook](block_hook.py) | Read or transform inputs before normalization, after normalization, or after attention and FFN |
| Delegate | [BaseDelegator](delegate.py) | Invoke the original block on the selected tokens and control how its output is restored |

[ShivaDitBlockHook](../shiva_hook.py) uses delegate mode. It selects token indices, gathers the shorter sequence and any matching rotary position embeddings, invokes the block, and restores the original sequence layout. Review [the benchmark guide](../../benchmark/README.md) for loading a checkpoint and running inference.
