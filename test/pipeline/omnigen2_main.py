import torch
from PIL import Image
from shiva_dit.registry.pipelines import OmniGen2Pipeline
from shiva_dit.registry.models import OmniGen2Transformer2DModel

model_path = '../OmniGen2'

pipe = OmniGen2Pipeline.from_pretrained(
    model_path, torch_dtype=torch.bfloat16
)
pipe.transformer = OmniGen2Transformer2DModel.from_pretrained(
    model_path,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)
pipe = pipe.to('cuda')

image = Image.open("./pic/two_man.jpg").convert("RGB")

## Multi-modal to Image
# In the prompt, we use the placeholder to represent the image. The image placeholder should be in the format of <img><|image_*|></img>
# You can add multiple images in the input_images. Please ensure that each image has its placeholder. For example, for the list input_images [img1_path, img2_path], the prompt needs to have two placeholders: <img><|image_1|></img>, <img><|image_2|></img>.
image = pipe(
    prompt="A man in a black shirt is reading a book. The man is the left man in the first image.",
    input_images=[image],
    height=1024,
    width=1024,
    num_inference_steps=50,
    text_guidance_scale=5.0,
    image_guidance_scale=2.0,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]
image.save("omnigen2_main.png")  # save output PIL image
