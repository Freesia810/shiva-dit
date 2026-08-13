import torch
from diffusers.pipelines import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "../stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
).to("mps")

pipe.vae.to(torch.float32)

# Convert latents to fp32 before VAE decoding.
original_vae_decode = pipe.vae.decode


def patched_decode(z, *args, **kwargs):
    return original_vae_decode(z.float(), *args, **kwargs)


pipe.vae.decode = patched_decode

image = pipe(
    "A cat holding a sign that says hello world",
    negative_prompt="",
    num_inference_steps=28,
    guidance_scale=7.0,
).images[0]
image.save("sd3.png")
