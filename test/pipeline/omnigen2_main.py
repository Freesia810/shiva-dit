import torch
from PIL import Image
from shiva_dit.registry.pipelines import OmniGen2Pipeline

pipe = OmniGen2Pipeline.from_pretrained(
    "../OmniGen2", torch_dtype=torch.bfloat16
).to("cuda")

image = Image.open("./pic/two_man.jpg").convert("RGB")

## Multi-modal to Image
# In the prompt, we use the placeholder to represent the image. The image placeholder should be in the format of <img><|image_*|></img>
# You can add multiple images in the input_images. Please ensure that each image has its placeholder. For example, for the list input_images [img1_path, img2_path], the prompt needs to have two placeholders: <img><|image_1|></img>, <img><|image_2|></img>.
image = pipe(
    prompt="A man in a black shirt is reading a book. The man is the left man in <img><|image_1|></img>.",
    input_images=[image],
    height=1024,
    width=1024,
    text_guidance_scale=4.0,
    image_guidance_scale=1.0,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]
image.save("omnigen_main.png")  # save output PIL image
