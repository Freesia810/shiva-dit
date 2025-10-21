import torch
from typing import Literal
from project_name.hook import BaseDiTBlockHook,HookManager


class PrintDiTBlockHook(BaseDiTBlockHook):
    def __init__(self):
        super().__init__()

    def close(self):
        print(f'TestDiTBlockHook[{self.layer_name}] closed.')

    def hook_pre_norm(
        self,
        module_name: Literal["self-attn", "cross-attn", "ffn", "parallel"],
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        temb: torch.Tensor = None,
    ):
        print(f"{self.layer_name}-{module_name} prenorm [{hidden_states.shape}], [{encoder_hidden_states.shape if encoder_hidden_states is not None else 'None'}]")
        return hidden_states, encoder_hidden_states
    
    def hook_after_norm(
        self, 
        module_name: Literal["self-attn", "cross-attn", "ffn"],
        norm_hidden_states: torch.Tensor,
        norm_encoder_hidden_states: torch.Tensor = None,
        temb: torch.Tensor = None,
    ):
        print(f"{self.layer_name}-{module_name} afternorm [{norm_hidden_states.shape}], [{norm_encoder_hidden_states.shape if norm_encoder_hidden_states is not None else 'None'}]")
        return norm_hidden_states, norm_encoder_hidden_states
    
    def hook_after_module(
        self, 
        module_name: Literal["self-attn", "cross-attn", "ffn"],
        hidden_output: torch.Tensor,
        encoder_hidden_output: torch.Tensor = None,
    ):
        print(f"{self.layer_name}-{module_name} aftermodule [{hidden_output.shape}], [{encoder_hidden_output.shape if encoder_hidden_output is not None else 'None'}]")
        return hidden_output, encoder_hidden_output

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
        print(f"{self.layer_name}-{module_name} pre attn")
        return hidden_states_qkv, encoder_hidden_states_qkv

    def hook_after_attn(
        self,
        module_name: Literal["self-attn", "cross-attn"],
        attn_weights: torch.Tensor,
        encoder_attn_weights: torch.Tensor = None,
    ):
        print(f"{self.layer_name}-{module_name} after attn [{attn_weights.shape}], [{encoder_attn_weights.shape if encoder_attn_weights is not None else 'None'}]")
        return attn_weights, encoder_attn_weights


def test_print_dit_block_hook():
    from diffusers.pipelines import StableDiffusion3Pipeline
    from diffusers.pipelines import FluxPipeline
    from diffusers.pipelines import QwenImagePipeline
    from diffusers.pipelines import PixArtSigmaPipeline
    from diffusers.pipelines import WanPipeline
    from diffusers.models.autoencoders import AutoencoderKLWan
    from diffusers.utils import export_to_video, load_image

    # model_type = 'sd3'
    # PipelineType = StableDiffusion3Pipeline
    # dtype = torch.float16
    # model_path = "../stable-diffusion-3-medium-diffusers"

    # model_type = 'flux-dev'
    # PipelineType = FluxPipeline
    # dtype = torch.bfloat16
    # model_path = "../FLUX.1-dev"

    # model_type = 'qwen-image'
    # PipelineType = QwenImagePipeline
    # dtype = torch.bfloat16
    # model_path = "../Qwen-Image"

    # model_type = 'pixart-sigma'
    # PipelineType = PixArtSigmaPipeline
    # dtype = torch.float16
    # model_path = "../PixArt-Sigma-XL-2-1024-MS"

    # pipe = PipelineType.from_pretrained(model_path, torch_dtype=dtype, use_safetensors=True).to("cuda")
    # model_attrs = ('transformer')

    model_type = 'wan'
    PipelineType = WanPipeline
    dtype = torch.bfloat16
    model_path = "../Wan2.2-T2V-A14B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
    pipe = PipelineType.from_pretrained(model_path, torch_dtype=dtype, vae=vae).to("cuda")
    model_attrs = ('transformer', 'transformer_2')

    HookManager.configure_hooks(
        model_type=model_type,
        default=(PrintDiTBlockHook, {}),
    )
    with HookManager.enabled():
        HookManager.inject_hooks(model_type=model_type, pipe=pipe, model_attrs=model_attrs)
        if model_type == 'wan':
            height = 720
            width = 1280
            prompt = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
            negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=81,
                guidance_scale=4.0,
                guidance_scale_2=3.0,
                num_inference_steps=2,
            ).frames[0]
            export_to_video(output, f"{model_type}_with_hooks.mp4", fps=16)
        else:
            if model_type == 'sd3':
                prompt = "A cat holding a sign that says hello worl"
                image = pipe(
                    prompt,
                    negative_prompt="",
                    num_inference_steps=28,
                    guidance_scale=7.0,
                ).images[0]
            elif model_type == 'flux-dev':
                prompt = "A cat holding a sign that says hello world"
                image = pipe(
                    prompt,
                    height=1024,
                    width=1024,
                    guidance_scale=3.5,
                    num_inference_steps=50,
                    max_sequence_length=512,
                    generator=torch.Generator("cpu").manual_seed(0)
                ).images[0]
            elif model_type == 'qwen-image':
                positive_magic = {
                    "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
                    "zh": ", 超清，4K，电影级构图." # for chinese prompt
                }
                # Generate image
                prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition'''
                negative_prompt = " " # using an empty string if you do not have specific concept to remove
                # Generate with different aspect ratios
                aspect_ratios = {
                    "1:1": (1328, 1328),
                    "16:9": (1664, 928),
                    "9:16": (928, 1664),
                    "4:3": (1472, 1140),
                    "3:4": (1140, 1472),
                    "3:2": (1584, 1056),
                    "2:3": (1056, 1584),
                }
                width, height = aspect_ratios["16:9"]
                image = pipe(
                    prompt=prompt + positive_magic["en"],
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=10,
                    true_cfg_scale=4.0,
                    generator=torch.Generator(device="cuda").manual_seed(42)
                ).images[0]
            elif model_type == 'pixart-sigma':
                prompt = "A small cactus with a happy face in the Sahara desert."
                image = pipe(prompt).images[0]
            
            image.save(f"{model_type}_with_hooks.png")

if __name__ == "__main__":
    test_print_dit_block_hook()
