import torch
from diffusers.pipelines import WanPipeline
from diffusers.models.autoencoders import AutoencoderKLWan
from diffusers.utils import export_to_video

dtype = torch.bfloat16
device = "cuda:0"
vae = AutoencoderKLWan.from_pretrained(
    "../Wan2.2-T2V-A14B-Diffusers", subfolder="vae", torch_dtype=torch.float32
)
pipe = WanPipeline.from_pretrained(
    "../Wan2.2-T2V-A14B-Diffusers", vae=vae, torch_dtype=dtype
)
pipe.to(device)

height = 720
width = 1280

prompt = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
negative_prompt = "overexposed, static, blurry details, subtitles, washed out, low quality, JPEG artifacts, malformed hands, malformed face, fused fingers, cluttered background"
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
