import torch
from diffusers.pipelines import OmniGenPipeline

pipe = OmniGenPipeline.from_pretrained("../OmniGen-v1-diffusers", torch_dtype=torch.bfloat16).to("cuda")
# Note: Your local model path is also acceptable, such as 'pipe = OmniGenPipeline.from_pretrained(your_local_model_path)', where all files in your_local_model_path should be organized as https://huggingface.co/Shitao/OmniGen-v1/tree/main


## Multi-modal to Image
# In the prompt, we use the placeholder to represent the image. The image placeholder should be in the format of <img><|image_*|></img>
# You can add multiple images in the input_images. Please ensure that each image has its placeholder. For example, for the list input_images [img1_path, img2_path], the prompt needs to have two placeholders: <img><|image_1|></img>, <img><|image_2|></img>.
image = pipe(
    prompt="A man in a black shirt is reading a book. The man is the left man in <img><|image_1|></img>.",
    input_images=["./pic/two_man.jpg"],
    height=1024, 
    width=1024,
    guidance_scale=2.5, 
    img_guidance_scale=1.6,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]
image.save("omnigen_main.png")  # save output PIL image
