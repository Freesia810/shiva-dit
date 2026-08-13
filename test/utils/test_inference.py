import torch
from shiva_dit.utils.inference_utils import load_pipeline_for_inference

CKPT_PATH = "./checkpoints/best.pt"
BASE_MODEL_PATH = None
DEVICE = "cuda"
DTYPE = torch.bfloat16
PROMPT = "A beautiful painting of a cat."
OUTPUT_IMAGE_PATH = "output_image.png"


@torch.no_grad()
def main():
    pipe = load_pipeline_for_inference(
        ckpt_path=CKPT_PATH, base_model_path=BASE_MODEL_PATH, device=DEVICE, dtype=DTYPE
    )

    print(f"Generating image for prompt: '{PROMPT}'")
    generator = torch.Generator(device=DEVICE).manual_seed(42)

    image = pipe(
        prompt=PROMPT,
        generator=generator,
        num_inference_steps=30,
    ).images[0]

    # Save the result.
    image.save(OUTPUT_IMAGE_PATH)
    print(f"Image saved to {OUTPUT_IMAGE_PATH}")


if __name__ == "__main__":
    main()
