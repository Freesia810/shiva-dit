import torch
import numpy as np
from diffusers.pipelines import WanPipeline
from diffusers.models.autoencoders import AutoencoderKLWan
from diffusers.utils import export_to_video, load_image

dtype = torch.bfloat16
device = "cuda:0"
vae = AutoencoderKLWan.from_pretrained("../Wan2.2-T2V-A14B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained("../Wan2.2-T2V-A14B-Diffusers", vae=vae, torch_dtype=dtype)
pipe.to(device)

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
    num_inference_steps=40,
).frames[0]
export_to_video(output, "t2v_out.mp4", fps=16)
