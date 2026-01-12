import torch
from typing import Literal, Tuple, Union, Optional, List
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
    OmniGenPipeline,
    WanPipeline,
)

default_model_config = {
    "sd3": ("../stable-diffusion-3-medium-diffusers", torch.float16, StableDiffusion3Pipeline),
    "flux-dev": ("../FLUX.1-dev", torch.bfloat16, FluxPipeline),
    "pixart-sigma": ("../PixArt-Sigma-XL-2-1024-MS", torch.float16, PixArtSigmaPipeline),
    "qwen-image": ("../Qwen-Image", torch.bfloat16, QwenImagePipeline),
    "flux-kontext": ("../FLUX.1-Kontext-dev", torch.bfloat16, FluxKontextPipeline),
    "qwen-image-edit": ("../Qwen-Image-Edit", torch.bfloat16, QwenImageEditPipeline),
    "omnigen": ("../OmniGen-v1-diffusers", torch.bfloat16, OmniGenPipeline),
    "wan": ("../Wan2.2-T2V-A14B-Diffusers", torch.bfloat16, WanPipeline),
}

def get_pipeline(model_type: Literal["sd3", "flux-dev", "pixart-sigma", "qwen-image", "flux-kontext", "qwen-image-edit", "omnigen", "wan"], model_path: str = None, dtype: torch.dtype = None, device: torch.device = None):
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
        kwargs.setdefault('num_inference_steps', 50)
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
    elif isinstance(pipeline, OmniGenPipeline):
        input_image = Image.open("pic/two_man.jpg").convert("RGB")
        kwargs.setdefault('input_images', [input_image])
        kwargs.setdefault('prompt', "A man in a black shirt is reading a book. The man is the right man in <img><|image_1|></img>.")
        kwargs.setdefault('height', 1024)
        kwargs.setdefault('width', 1024)
        kwargs.setdefault('num_inference_steps', 50)
        kwargs.setdefault('guidance_scale', 2.5)
        kwargs.setdefault('img_guidance_scale', 1.6)
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

def _slice_rope(rope: Tuple[torch.Tensor, torch.Tensor], start: int, end: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    return tuple(t[start:end] for t in rope)

def parse_image_rotary_emb(
    model_type: Literal["sd3", "flux-dev", "pixart-sigma", "qwen-image", "flux-kontext", "qwen-image-edit", "omnigen", "wan"],
    image_rotary_emb: any,
    hidden_seq_len: int,
    encoder_hidden_seq_len: int,
    split_extra: bool = True
):
    # === Flux Family Logic (Unified) ===
    if model_type in ("flux-dev", "flux-kontext"):
        # image_rotary_emb is strictly (cos, sin)
        # Structure: [Encoder | Hidden | Extra(optional)]

        encoder_rotary_emb = _slice_rope(image_rotary_emb, 0, encoder_hidden_seq_len)
        
        if model_type == "flux-dev":
            hidden_rotary_emb = _slice_rope(image_rotary_emb, encoder_hidden_seq_len, None)
            return hidden_rotary_emb, encoder_rotary_emb, None
        
        else: # flux-kontext
            if split_extra:
                h_end = encoder_hidden_seq_len + hidden_seq_len
                hidden_rotary_emb = _slice_rope(image_rotary_emb, encoder_hidden_seq_len, h_end)
                extra_hidden_rotary_emb = _slice_rope(image_rotary_emb, h_end, None)
            else:
                hidden_rotary_emb = _slice_rope(image_rotary_emb, encoder_hidden_seq_len, None)
                extra_hidden_rotary_emb = None
            
            return hidden_rotary_emb, encoder_rotary_emb, extra_hidden_rotary_emb
    
    # === OmniGen Family Logic ===
    # we assume time_seq as extra, for not changing the signature
    # we shouldn't change the temb_rotary_emb
    elif model_type in ("omnigen"):
        # image_rotary_emb is strictly (cos, sin)
        # Structure: [Encoder | Time | Hidden]
        time_seq_len = image_rotary_emb[0].shape[0] - encoder_hidden_seq_len - hidden_seq_len 

        encoder_rotary_emb = _slice_rope(image_rotary_emb, 0, encoder_hidden_seq_len)
        temb_rotary_emb = _slice_rope(image_rotary_emb, encoder_hidden_seq_len, encoder_hidden_seq_len + time_seq_len)
        
        h_start = encoder_hidden_seq_len + time_seq_len
        h_end = encoder_hidden_seq_len + time_seq_len + hidden_seq_len
        hidden_rotary_emb = _slice_rope(image_rotary_emb, h_start, h_end)

        return hidden_rotary_emb, encoder_rotary_emb, temb_rotary_emb

    # === Qwen Family Logic ===
    elif model_type == "qwen-image":
        # Input: (HiddenPart, EncoderPart)
        hidden_emb, encoder_emb = image_rotary_emb
        
        if not isinstance(hidden_emb, tuple):
            hidden_emb = (hidden_emb,)
        if not isinstance(encoder_emb, tuple):
            encoder_emb = (encoder_emb,)
            
        return hidden_emb, encoder_emb, None

    elif model_type == "qwen-image-edit":
        # Input: (Hidden+Extra, Encoder)
        hidden_complex, encoder_emb = image_rotary_emb
        
        if not isinstance(encoder_emb, tuple):
            encoder_emb = (encoder_emb,)

        is_tuple_input = isinstance(hidden_complex, tuple)
        
        if split_extra:
            if is_tuple_input:
                hidden_rotary_emb = _slice_rope(hidden_complex, 0, hidden_seq_len)
                extra_hidden_rotary_emb = _slice_rope(hidden_complex, hidden_seq_len, None)
            else:
                hidden_rotary_emb = (hidden_complex[:hidden_seq_len, ...],)
                extra_hidden_rotary_emb = (hidden_complex[hidden_seq_len:, ...],)
        else:
            hidden_rotary_emb = hidden_complex if is_tuple_input else (hidden_complex,)
            extra_hidden_rotary_emb = None

        return hidden_rotary_emb, encoder_emb, extra_hidden_rotary_emb

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def join_image_rotary_emb(
    model_type: Literal["sd3", "flux-dev", "pixart-sigma", "qwen-image", "flux-kontext", "qwen-image-edit", "omnigen", "wan"],
    hidden_rotary_emb: Tuple[torch.Tensor, ...],
    encoder_rotary_emb: Tuple[torch.Tensor, ...],
    extra_hidden_rotary_emb: Optional[Tuple[torch.Tensor, ...]] = None
):
    # === Flux Family Logic ===
    if model_type in ('flux-dev', 'flux-kontext'):
        parts_cos = [encoder_rotary_emb[0], hidden_rotary_emb[0]]
        parts_sin = [encoder_rotary_emb[1], hidden_rotary_emb[1]]
        
        if model_type == 'flux-kontext' and extra_hidden_rotary_emb is not None:
            parts_cos.append(extra_hidden_rotary_emb[0])
            parts_sin.append(extra_hidden_rotary_emb[1])
            
        return (torch.cat(parts_cos, dim=0), torch.cat(parts_sin, dim=0))
    
    # === OmniGen Family Logic ===
    elif model_type == 'omnigen':
        # extra is time
        parts_cos = [encoder_rotary_emb[0], extra_hidden_rotary_emb[0], hidden_rotary_emb[0]]
        parts_sin = [encoder_rotary_emb[1], extra_hidden_rotary_emb[1], hidden_rotary_emb[1]]
        return (torch.cat(parts_cos, dim=0), torch.cat(parts_sin, dim=0))

    # === Qwen Family Logic ===
    elif model_type == 'qwen-image':
        if len(hidden_rotary_emb) == 1:
            return (hidden_rotary_emb[0], encoder_rotary_emb[0])
        return (hidden_rotary_emb, encoder_rotary_emb)

    elif model_type == 'qwen-image-edit':
        if extra_hidden_rotary_emb is not None:
            if len(hidden_rotary_emb) == 1:
                merged_hidden = torch.cat([hidden_rotary_emb[0], extra_hidden_rotary_emb[0]], dim=0)
                final_hidden = (merged_hidden,)
            else:
                cos = torch.cat([hidden_rotary_emb[0], extra_hidden_rotary_emb[0]], dim=0)
                sin = torch.cat([hidden_rotary_emb[1], extra_hidden_rotary_emb[1]], dim=0)
                final_hidden = (cos, sin)
        else:
            final_hidden = hidden_rotary_emb

        if len(final_hidden) == 1:
            return (final_hidden[0], encoder_rotary_emb[0])
        else:
            return (final_hidden, encoder_rotary_emb)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
