import torch
from typing import Literal, Tuple
from PIL import Image
from diffusers import DiffusionPipeline
from diffusers.utils import load_image
from diffusers.models.autoencoders import AutoencoderKLWan
from diffusers.pipelines import (
    StableDiffusion3Pipeline,
    FluxPipeline,
    PixArtSigmaPipeline,
    QwenImagePipeline,
    FluxKontextPipeline,
    QwenImageEditPipeline,
    WanPipeline,
)

default_model_config = {
    "sd3": ("../stable-diffusion-3-medium-diffusers", torch.float16, StableDiffusion3Pipeline),
    "flux-dev": ("../FLUX.1-dev", torch.bfloat16, FluxPipeline),
    "pixart-sigma": ("../PixArt-Sigma-XL-2-1024-MS", torch.float16, PixArtSigmaPipeline),
    "qwen-image": ("../Qwen-Image", torch.bfloat16, QwenImagePipeline),
    "flux-kontext": ("../FLUX.1-Kontext-dev", torch.bfloat16, FluxKontextPipeline),
    "qwen-image-edit": ("../Qwen-Image-Edit", torch.bfloat16, QwenImageEditPipeline),
    "wan": ("../Wan2.2-T2V-A14B-Diffusers", torch.bfloat16, WanPipeline),
}

def get_pipeline(model_type: Literal["sd3", "flux-dev", "pixart-sigma", "qwen-image", "flux-kontext", "qwen-image-edit", "wan"], model_path: str = None, dtype: torch.dtype = None, device: torch.device = None):
    model_path = model_path or default_model_config[model_type][0]
    dtype = dtype or default_model_config[model_type][1]
    PipelineType = default_model_config[model_type][2]
    device = device or torch.device("cuda")

    if model_type == 'wan':
        vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
        pipe = PipelineType.from_pretrained(model_path, torch_dtype=dtype, vae=vae).to(device)
        model_attrs = ('transformer', 'transformer_2')
    else:
        pipe = PipelineType.from_pretrained(model_path, torch_dtype=dtype, use_safetensors=True).to(device)
        model_attrs = ('transformer',)
    
    return pipe, model_attrs

def process_quick_start(pipeline: DiffusionPipeline, **kwargs):
    if isinstance(pipeline, StableDiffusion3Pipeline):
        kwargs.setdefault('prompt', "A cat holding a sign that says hello world")
        kwargs.setdefault('negative_prompt', "")
        kwargs.setdefault('num_inference_steps', 28)
        kwargs.setdefault('guidance_scale', 7.0)
        kwargs.setdefault('generator', torch.Generator(device="cuda").manual_seed(0))
        return pipeline(**kwargs).images[0]
    elif isinstance(pipeline, FluxPipeline):
        kwargs.setdefault('prompt', "A cat holding a sign that says hello world")
        kwargs.setdefault('height', 1024)
        kwargs.setdefault('width', 1024)
        kwargs.setdefault('guidance_scale', 3.5)
        kwargs.setdefault('num_inference_steps', 50)
        kwargs.setdefault('max_sequence_length', 512)
        kwargs.setdefault('generator', torch.Generator("cpu").manual_seed(0))
        return pipeline(**kwargs).images[0]
    elif isinstance(pipeline, PixArtSigmaPipeline):
        kwargs.setdefault('prompt', "A small cactus with a happy face in the Sahara desert.")
        kwargs.setdefault('generator', torch.Generator(device="cuda").manual_seed(0))
        return pipeline(**kwargs).images[0]
    elif isinstance(pipeline, QwenImagePipeline):
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

        kwargs.setdefault('prompt', prompt + positive_magic["en"])
        kwargs.setdefault('negative_prompt', negative_prompt)
        kwargs.setdefault('width', width)
        kwargs.setdefault('height', height)
        kwargs.setdefault('num_inference_steps', 10)
        kwargs.setdefault('true_cfg_scale', 4.0)
        kwargs.setdefault('generator', torch.Generator(device="cuda").manual_seed(42))
        return pipeline(**kwargs).images[0]
    elif isinstance(pipeline, FluxKontextPipeline):
        input_image = load_image("pic/cat.png")
        kwargs.setdefault('image', input_image)
        kwargs.setdefault('prompt', "Add a hat to the cat")
        kwargs.setdefault('guidance_scale', 2.5)
        kwargs.setdefault('generator', torch.Generator(device="cuda").manual_seed(0))
        return pipeline(**kwargs).images[0]
    elif isinstance(pipeline, QwenImageEditPipeline):
        input_image = Image.open("pic/rabbit.png").convert("RGB")
        kwargs.setdefault('image', input_image)
        kwargs.setdefault('prompt', "Change the rabbit's color to purple, with a flash light background.")
        kwargs.setdefault('negative_prompt', " ")
        kwargs.setdefault('true_cfg_scale', 4.0)
        kwargs.setdefault('num_inference_steps', 50)
        kwargs.setdefault('height', 1024)
        kwargs.setdefault('width', 1024)
        kwargs.setdefault('generator', torch.Generator(device="cuda").manual_seed(0))
        return pipeline(**kwargs).images[0]
    elif isinstance(pipeline, WanPipeline):
        kwargs.setdefault('prompt', "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.")
        kwargs.setdefault('negative_prompt', "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
        kwargs.setdefault('height', 720)
        kwargs.setdefault('width', 1280)
        kwargs.setdefault('num_frames', 81)
        kwargs.setdefault('guidance_scale', 4.0)
        kwargs.setdefault('guidance_scale_2', 3.0)
        kwargs.setdefault('num_inference_steps', 40)
        return pipeline(**kwargs).frames[0]

def parse_image_rotary_emb(model_type: Literal["sd3", "flux-dev", "pixart-sigma", "qwen-image", "flux-kontext", "qwen-image-edit", "wan"], image_rotary_emb: any, hidden_seq_len, encoder_hidden_seq_len, split_extra: bool = True):
    ## flux-dev ([encoder + hidden], [encoder + hidden])
    ## flux-kontext ([encoder + hidden + extra_hidden], [encoder + hidden + extra_hidden])
    ## qwen-image (([hidden], [hidden]), ([encoder], [encoder])) or ([hidden], [encoder])
    ## qwen-image-edit (([hidden + extra_hidden], [hidden + extra_hidden]), ([encoder] + [encoder])) or ([hidden + extra_hidden], [encoder])

    if model_type == "flux-dev":
        # ([encoder + hidden], [encoder + hidden])
        cos, sin = image_rotary_emb
        hidden_rotary_emb = (cos[encoder_hidden_seq_len:, ...], sin[encoder_hidden_seq_len:, ...])
        encoder_rotary_emb = (cos[:encoder_hidden_seq_len, ...], sin[:encoder_hidden_seq_len, ...])
        return hidden_rotary_emb, encoder_rotary_emb, None
    elif model_type == "flux-kontext":
        # ([encoder + hidden + extra_hidden], [encoder + hidden + extra_hidden])
        cos, sin = image_rotary_emb
        encoder_rotary_emb = (
            cos[:encoder_hidden_seq_len, ...],
            sin[:encoder_hidden_seq_len, ...],
        )
        if split_extra:
            hidden_rotary_emb = (
                cos[encoder_hidden_seq_len:encoder_hidden_seq_len + hidden_seq_len, ...],
                sin[encoder_hidden_seq_len:encoder_hidden_seq_len + hidden_seq_len, ...],
            )
            extra_hidden_rotary_emb = (
                cos[encoder_hidden_seq_len + hidden_seq_len:, ...],
                sin[encoder_hidden_seq_len + hidden_seq_len:, ...],
            )
        else:
            hidden_rotary_emb = (
                cos[encoder_hidden_seq_len:, ...],
                sin[encoder_hidden_seq_len:, ...],
            )
            extra_hidden_rotary_emb = None
        return hidden_rotary_emb, encoder_rotary_emb, extra_hidden_rotary_emb
    elif model_type == "qwen-image":
        # (([hidden], [hidden]), ([encoder], [encoder])) or ([hidden], [encoder])
        hidden_emb, encoder_emb = image_rotary_emb
        if not isinstance(hidden_emb, tuple):
            hidden_emb = (hidden_emb,)
            encoder_emb = (encoder_emb,)
        return hidden_emb, encoder_emb, None
    elif model_type == "qwen-image-edit":
        # (([hidden + extra_hidden], [hidden + extra_hidden]), ([encoder], [encoder])) or ([hidden + extra_hidden], [encoder])
        hidden_emb, encoder_emb = image_rotary_emb
        if not isinstance(hidden_emb, tuple):
            encoder_emb = (encoder_emb,)
            if split_extra:
                hidden_rotary_emb = (hidden_emb[:hidden_seq_len, ...],)
                extra_hidden_rotary_emb = (hidden_emb[hidden_seq_len:, ...],)
            else:
                hidden_rotary_emb = (hidden_emb,)
                extra_hidden_rotary_emb = None
        else:
            hidden_rotary_emb_cos, hidden_rotary_emb_sin = hidden_emb
            if split_extra:
                hidden_rotary_emb = (hidden_rotary_emb_cos[:hidden_seq_len, ...], hidden_rotary_emb_sin[:hidden_seq_len, ...])
                extra_hidden_rotary_emb = (hidden_rotary_emb_cos[hidden_seq_len:, ...], hidden_rotary_emb_sin[hidden_seq_len:, ...])
            else:
                hidden_rotary_emb = (hidden_rotary_emb_cos, hidden_rotary_emb_sin)
                extra_hidden_rotary_emb = None
        return hidden_rotary_emb, encoder_emb, extra_hidden_rotary_emb
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def join_image_rotary_emb(model_type: Literal["sd3", "flux-dev", "pixart-sigma", "qwen-image", "flux-kontext", "qwen-image-edit", "wan"], hidden_rotary_emb: Tuple[torch.Tensor, torch.Tensor], encoder_rotary_emb: Tuple[torch.Tensor, torch.Tensor], extra_hidden_rotary_emb: Tuple[torch.Tensor, torch.Tensor] = None):
    if model_type == 'flux-dev':
        cos = torch.cat([encoder_rotary_emb[0], hidden_rotary_emb[0]], dim=0)
        sin = torch.cat([encoder_rotary_emb[1], hidden_rotary_emb[1]], dim=0)
        return (cos, sin)
    elif model_type == 'flux-kontext':
        if extra_hidden_rotary_emb is None:
            cos = torch.cat([encoder_rotary_emb[0], hidden_rotary_emb[0]], dim=0)
            sin = torch.cat([encoder_rotary_emb[1], hidden_rotary_emb[1]], dim=0)
            return (cos, sin)
        else:
            cos = torch.cat([encoder_rotary_emb[0], hidden_rotary_emb[0], extra_hidden_rotary_emb[0]], dim=0)
            sin = torch.cat([encoder_rotary_emb[1], hidden_rotary_emb[1], extra_hidden_rotary_emb[1]], dim=0)
            return (cos, sin)
    elif model_type == 'qwen-image':
        if len(hidden_rotary_emb) == 1:
            return (hidden_rotary_emb[0], encoder_rotary_emb[0])
        return (hidden_rotary_emb, encoder_rotary_emb)
    elif model_type == 'qwen-image-edit':
        if len(hidden_rotary_emb) == 1:
            if extra_hidden_rotary_emb is None:
                return (hidden_rotary_emb[0], encoder_rotary_emb[0])
            else:
                hidden_rotary_emb = torch.cat([hidden_rotary_emb[0], extra_hidden_rotary_emb[0]], dim=0)
                return (hidden_rotary_emb, encoder_rotary_emb[0])
        else:
            if extra_hidden_rotary_emb is None:
                return (hidden_rotary_emb, encoder_rotary_emb)
            else:
                cos = torch.cat([hidden_rotary_emb[0], extra_hidden_rotary_emb[0]], dim=0)
                sin = torch.cat([hidden_rotary_emb[1], extra_hidden_rotary_emb[1]], dim=0)
                return ((cos, sin), encoder_rotary_emb)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
