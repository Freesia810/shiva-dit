import torch
from diffusers.pipelines.flux import FluxPipeline
from diffusers.pipelines.stable_diffusion_3 import StableDiffusion3Pipeline

pipe = FluxPipeline.from_pretrained("../FLUX.1-dev", dtype=torch.bfloat16).to("cuda")
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
image.save("flux-dev.png")
