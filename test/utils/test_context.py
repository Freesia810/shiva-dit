import torch
from PIL import Image
from shiva_dit.utils import global_context
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

def test_global_context(model_type):
    if model_type == 'sd3':
        PipelineType = StableDiffusion3Pipeline
        dtype = torch.float16
        model_path = "../stable-diffusion-3-medium-diffusers"
    elif model_type == 'flux-dev':
        PipelineType = FluxPipeline
        dtype = torch.bfloat16
        model_path = "../FLUX.1-dev"
    elif model_type == 'pixart-sigma':
        PipelineType = PixArtSigmaPipeline
        dtype = torch.float16
        model_path = "../PixArt-Sigma-XL-2-1024-MS"
    elif model_type == 'qwen-image':
        PipelineType = QwenImagePipeline
        dtype = torch.bfloat16
        model_path = "../Qwen-Image"
    elif model_type == 'flux-kontext':
        PipelineType = FluxKontextPipeline
        dtype = torch.bfloat16
        model_path = "../FLUX.1-Kontext-dev"
    elif model_type == 'qwen-image-edit':
        PipelineType = QwenImageEditPipeline
        dtype = torch.bfloat16
        model_path = "../Qwen-Image-Edit"
    elif model_type == 'wan':
        PipelineType = WanPipeline
        dtype = torch.bfloat16
        model_path = "../Wan2.2-T2V-A14B-Diffusers"
    else:
        raise ValueError(f"Unknown model type {model_type}")

    pipe = PipelineType.from_pretrained(
        model_path, torch_dtype=dtype, use_safetensors=True
    ).to("cuda")

    if model_type == "sd3":
        prompt = "A cat holding a sign that says hello worl"
        image = pipe(
            prompt,
            negative_prompt="",
            num_inference_steps=28,
            guidance_scale=7.0,
            height=512,
            width=1024,
        ).images[0]
    elif model_type == "flux-dev":
        prompt = "A cat holding a sign that says hello world"
        image = pipe(
            prompt,
            height=512,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(0),
        ).images[0]
    elif model_type == "pixart-sigma":
        prompt = "A small cactus with a happy face in the Sahara desert."
        image = pipe(
            prompt,
            height=1024,
            width=2048,
        ).images[0]
    elif model_type == "qwen-image":
        positive_magic = {
            "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
            "zh": ", 超清，4K，电影级构图." # for chinese prompt
        }

        prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition'''
        negative_prompt = " " # using an empty string if you do not have specific concept to remove
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
            height=height,
            width=width,
            num_inference_steps=50,
            true_cfg_scale=4.0,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).images[0]
    elif model_type == "flux-kontext":
        input_image = load_image("pic/cat.png")
        image = pipe(
            image=input_image,
            prompt="Add a hat to the cat",
            guidance_scale=2.5
        ).images[0]
    elif model_type == "qwen-image-edit":
        input_image = Image.open("pic/rabbit.png").convert("RGB")
        prompt = "Change the rabbit's color to purple, with a flash light background."
        image = pipe(
            image=input_image,
            prompt=prompt,
            height=1024,
            width=1024,
            generator=torch.manual_seed(0),
            true_cfg_scale=4.0,
            negative_prompt=" ",
            num_inference_steps=50,
        ).images[0]

    print(global_context)
    image.save(f"{model_type}_with_context.png")


if __name__ == "__main__":
    test_global_context("qwen-image-edit")
