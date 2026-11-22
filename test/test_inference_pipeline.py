"""Check checkpoint and LUT compatibility with a tiny local SD3 pipeline."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel, StableDiffusion3Pipeline
from PIL import Image

from benchmark import evaluate, lut
from shiva_dit.hook import HookManager
from shiva_dit.shiva_hook import ShivaDitBlockHook
from shiva_dit.utils.context import global_context
from shiva_dit.utils.inference_utils import load_gater_lut, load_pipeline_for_inference


def tiny_pipeline(*args, **kwargs):
    torch.manual_seed(19)
    transformer = SD3Transformer2DModel(
        sample_size=4, patch_size=2, in_channels=4, out_channels=4, num_layers=2,
        attention_head_dim=8, num_attention_heads=2, joint_attention_dim=16,
        caption_projection_dim=16, pooled_projection_dim=16, pos_embed_max_size=4,
    )
    pipe = StableDiffusion3Pipeline(
        transformer=transformer, scheduler=FlowMatchEulerDiscreteScheduler(),
        vae=None, text_encoder=None, tokenizer=None, text_encoder_2=None,
        tokenizer_2=None, text_encoder_3=None, tokenizer_3=None,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe, ("transformer",)


def denoise(pipe, num_inference_steps=3, height=32, width=32, **kwargs):
    generator = torch.Generator().manual_seed(7)
    return pipe(
        prompt_embeds=torch.randn(1, 4, 16, generator=generator),
        pooled_prompt_embeds=torch.randn(1, 16, generator=generator),
        latents=torch.randn(1, 4, 4, 4, generator=generator),
        guidance_scale=1.0, output_type="latent",
        num_inference_steps=num_inference_steps, height=height, width=width,
    ).images


class InferencePipelineTests(unittest.TestCase):
    def test_sparse_backward_reaches_router_and_ratio_policy(self):
        pipe, model_attrs = tiny_pipeline()
        pipe.transformer.requires_grad_(False).train()
        config = {
            "num_router_groups": 1, "fixed_image_ratio": 0.5, "layer_embed_dim": 8,
            "bottleneck_dim": 8, "ratio_bottleneck_dim": 8, "skip_layer_indices": [0],
        }
        self.addCleanup(global_context.clear)
        HookManager.configure_hooks("sd3", default=(ShivaDitBlockHook, config))
        with HookManager.enabled():
            HookManager.inject_hooks("sd3", pipe, model_attrs)
            hook, = HookManager.get_all_hooks().values()
            global_context.update(
                current_stage_idx=1, current_timestep=torch.tensor(500),
                current_iteration=0, num_inference_steps=3,
            )
            output = pipe.transformer(
                hidden_states=torch.randn(2, 4, 4, 4),
                encoder_hidden_states=torch.randn(2, 4, 16),
                pooled_projections=torch.randn(2, 16),
                timestep=torch.tensor([500.0, 500.0]),
            ).sample
            output.square().mean().backward()
            for component in (hook.router, hook.image_ratio_policy):
                gradients = [p.grad for p in component.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients))
                self.assertGreater(sum(grad.abs().sum().item() for grad in gradients), 0)

    def test_checkpoint_and_lut_round_trip(self):
        self.addCleanup(HookManager.disable)
        meta = {
            "model_type": "sd3",
            "training_mode": "hook_only",
            "hook_info": {
                "class_name": "ShivaDitBlockHook",
                "config": {
                    "num_router_groups": 1, "fixed_image_ratio": 0.5,
                    "layer_embed_dim": 8, "bottleneck_dim": 8, "ratio_bottleneck_dim": 8,
                },
            },
        }
        with TemporaryDirectory() as directory, patch(
            "shiva_dit.utils.inference_utils.get_pipeline", side_effect=tiny_pipeline
        ):
            checkpoint = Path(directory) / "model.pt"
            lut_path = Path(directory) / "lut.pt"
            adaptive = load_pipeline_for_inference(meta_config=meta, device="cpu", dtype=torch.float32)
            expected = denoise(adaptive)
            with patch.object(lut, "process_quick_start", side_effect=denoise):
                table = lut.capture_lut(adaptive, list(HookManager.get_all_hooks().values()), 3, 32, 32)
            torch.save(table, lut_path)
            torch.testing.assert_close(table["ratio"][0], torch.ones(3))
            self.assertTrue(torch.all((table["ratio"][1] > 0) & (table["ratio"][1] < 1)))
            torch.save(meta | {"model": adaptive.transformer.state_dict()}, checkpoint)

            HookManager.disable()
            reloaded = load_pipeline_for_inference(
                str(checkpoint), device="cpu", dtype=torch.float32,
                gater_lut_path=lut_path, strict_checkpoint=True, expected_lut_steps=3,
            )
            actual = denoise(reloaded)
            self.assertTrue(torch.isfinite(actual).all())
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            self.assertNotIn("use_gater_lut", meta["hook_info"]["config"])
            with self.assertRaisesRegex(ValueError, "LUT contains 3 steps, requested 4"):
                load_pipeline_for_inference(gater_lut_path=lut_path, expected_lut_steps=4)

    def test_invalid_lut_shape_is_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "lut.pt"
            torch.save({"gater": torch.zeros(2, 3), "ratio": torch.zeros(3, 2)}, path)
            with self.assertRaisesRegex(ValueError, "same non-empty"):
                load_gater_lut(path)

    def test_evaluation_uses_the_same_subset_for_fid(self):
        class Evaluator:
            def __init__(self, **kwargs):
                self.prompts = []

            def update_fake(self, image, prompt):
                self.prompts.append(prompt)

            def compute(self, generated_images_dir):
                if self.prompts != ["first"]:
                    raise AssertionError(self.prompts)
                files = [path.name for path in Path(generated_images_dir).iterdir()]
                if files != ["p0000_s00.png"]:
                    raise AssertionError(files)
                return {"FID": 1.0}

        with TemporaryDirectory() as directory:
            path = Path(directory)
            prompts = path / "eval.jsonl"
            prompts.write_text('{"text":"first"}\n{"text":"second"}\n')
            for name in ("p0000_s00.png", "p0001_s00.png", "unrelated.png"):
                Image.new("RGB", (8, 8)).save(path / name)
            fid = path / "reference.npz"
            fid.touch()
            argv = [
                "benchmark.evaluate", "--image_dir", directory, "--jsonl_path", str(prompts),
                "--fid_stats", str(fid), "--limit", "1", "--device", "cpu",
            ]
            with patch.object(evaluate, "GenerationEvaluator", Evaluator), patch("sys.argv", argv):
                evaluate.main()
            self.assertTrue((path / "metrics.json").is_file())


if __name__ == "__main__":
    unittest.main()
