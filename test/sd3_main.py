import torch
from diffusers.pipelines.flux import FluxPipeline
from diffusers.pipelines.stable_diffusion_3 import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("../stable-diffusion-3-medium-diffusers", dtype=torch.float16).to("cuda")
image = pipe(
    "A cat holding a sign that says hello world",
    negative_prompt="",
    num_inference_steps=28,
    guidance_scale=7.0,
).images[0]
image.save("sd3.png")
