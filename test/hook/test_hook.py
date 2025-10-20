import torch
from typing import Literal
from project_name.utils.block_hook import BaseDiTBlockHook
from project_name.utils.hook_manager import HookManager


class PrintDiTBlockHook(BaseDiTBlockHook):
    def __init__(self):
        super().__init__()

    def close(self):
        print("TestDiTBlockHook closed.")

    def hook_pre_norm(
        self,
        module_name: Literal["self-attn", "cross-attn", "ffn", "parallel"],
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        temb: torch.Tensor = None,
    ):
        print(f"hook_pre_norm called for layer: {self.layer_name}-{module_name}")
        return hidden_states, encoder_hidden_states

    def hook_pre_attn(
        self,
        module_name: Literal["self-attn", "cross-attn"],
        hidden_states_qkv: tuple[
            torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
        ],
        encoder_hidden_states_qkv: tuple[
            torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
        ] = None,
    ):
        print(f"hook_pre_attn called for layer: {self.layer_name}-{module_name}")
        return hidden_states_qkv, encoder_hidden_states_qkv

    def hook_after_attn(
        self,
        module_name: Literal["self-attn", "cross-attn"],
        attn_weights: torch.Tensor,
        encoder_attn_weights: torch.Tensor = None,
    ):
        print(f"hook_after_attn called for layer: {self.layer_name}-{module_name}")
        return attn_weights, encoder_attn_weights


def test_print_dit_block_hook():
    from diffusers.pipelines import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained("../stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16).to("cuda")
    HookManager.configure_hooks(
        model_type="sd3",
        default=(PrintDiTBlockHook, {}),
    )
    with HookManager.enabled():
        HookManager.inject_hooks(model_type="sd3", model=pipe.transformer)
        image = pipe(
            "A cat holding a sign that says hello world",
            negative_prompt="",
            num_inference_steps=28,
            guidance_scale=7.0,
        ).images[0]
        image.save("sd3_with_hooks.png")

if __name__ == "__main__":
    test_print_dit_block_hook()
