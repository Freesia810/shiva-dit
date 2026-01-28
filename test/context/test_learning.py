import copy
import torch
from shiva_dit.utils import get_pipeline, process_quick_start
from PIL import Image
from shiva_dit.hook.hook_manager import HookManager
from shiva_dit.utils import global_context
from shiva_dit.shiva_hook import ShivaDitBlockHook


public_shiva_config = {
    "shiva_mode": "context",
    "context_ratio_mode": "cosine",
    "fixed_context_ratio": (1.0, 0.1),
    "context_pool_size": 64,

}

context_shared_config = {
    "qwen-image-edit": {
        "spatial_step_thresh": 0.2,
        "spatial_group_size": 2,
        "temporal_layer_thresh": 1.0,
        "temporal_interval": 4,
    }
}

def test_free_baseline(model_type, **kwargs):
    pipe, model_attrs = get_pipeline(model_type)
    free_config = copy.deepcopy(public_shiva_config)
    free_config['context_pruner_mode'] = "free"
    free_config["context_shared_config"] = context_shared_config[model_type]

    HookManager.configure_hooks(
        model_type=model_type,
        default=(ShivaDitBlockHook, free_config),
    )

    with HookManager.enabled():
        HookManager.inject_hooks(
            model_type=model_type, pipe=pipe, model_attrs=model_attrs
        )
        res = process_quick_start(pipe, **kwargs)
        res.save(f"{model_type}_free.png")

    print(global_context)


def test_learned_baseline(model_type, ckpt_path, **kwargs):
    pipe, model_attrs = get_pipeline(model_type)
    learned_config = copy.deepcopy(public_shiva_config)
    learned_config['context_pruner_mode'] = "learned"

    ckpt = torch.load(ckpt_path)
    assert ckpt['model_type'] == model_type, f"{ckpt['model_type']} != {model_type}"
    assert ckpt['hook_info']['class_name'] == 'ShivaDitBlockHook', f"{ckpt['hook_info']['class_name']} != ShivaDitBlockHook"
    assert ckpt['hook_info']['config']['context_pool_size'] == learned_config['context_pool_size'], f"{ckpt['hook_info']['context_pool_size']} != {learned_config['context_pool_size']}"
    learned_config['context_proj_dim'] = ckpt['hook_info']['config']['context_proj_dim']

    HookManager.configure_hooks(
        model_type=model_type,
        default=(ShivaDitBlockHook, learned_config),
    )

    with HookManager.enabled():
        HookManager.inject_hooks(
            model_type=model_type, pipe=pipe, model_attrs=model_attrs
        )

        # force load
        missing_keys, unexpected_keys = pipe.transformer.load_state_dict(ckpt['model'], strict=False)
        all_keys = set(pipe.transformer.state_dict().keys())
        num_loaded = len(all_keys) - len(missing_keys)
        print(f"✅ 成功加载参数项: {num_loaded}")
        print(f"❓ 缺失参数项 (Missing): {len(missing_keys)}")
        print(f"❗ 冗余参数项 (Unexpected): {len(unexpected_keys)}")

        res = process_quick_start(pipe, **kwargs)
        res.save(f"{model_type}_learned.png")

    print(global_context)

if __name__ == "__main__":
    max_sequence_length = None
    image = None
    num_images_per_prompt = 1


    # model_type = "flux-dev"
    # max_sequence_length = 512
    # prompt = "A race car driver stands in the foreground, his back to the viewer, clad in a dark blue racing suit adorned with sponsor logos including Puma, Red Bull, and various automotive brands, the fabric exhibiting a subtle sheen under the golden-hour light. His helmet, a glossy composite of deep red and black with intricate white and silver decals, features a reflective visor that captures the fiery hues of the setting sun, while the rear of his suit displays additional sponsor patches and a faint sheen of moisture from the humid track atmosphere. To his right, a sleek Formula 1 car dominates the midground, its aerodynamic body painted in dark blue with vibrant red accents and the Red Bull logo emblazoned on the nose and sidepods, the large rear tires kicking up a fine mist of water vapor as they grip the rain-slicked asphalt. The car’s carbon fiber chassis gleams with precision-engineered curves, and the cockpit reveals a glimpse of the driver’s gloved hands gripping the steering wheel, though the figure inside remains obscured by the halo device. The racetrack stretches into the distance, lined by multi-tiered grandstands packed with a dense crowd of spectators, their silhouettes blurred into a sea of color under the stadium lights, while the far background reveals a hazy city skyline with skyscrapers piercing the horizon beneath a sky ablaze with orange, crimson, and indigo clouds. The track surface glistens with recent rain, reflecting the fiery sky and the blurred shapes of distant race cars, some trailing plumes of white smoke as they accelerate down the straight, while the left grandstand curves into the distance, its upper tiers filled with spectators waving flags that flutter in the evening breeze. Airborne debris—shards of confetti or tire fragments—floats like embers in the warm light, adding a sense of chaotic energy to the scene. The sun, low on the horizon, bathes the scene in a warm, diffused glow, casting long shadows that stretch across the wet tarmac and illuminating the airborne particles with a golden halo. Subsurface scattering on the driver’s helmet visor reveals a faint glow where sunlight penetrates the composite material, while the shadows beneath the car’s chassis are rendered with deep, cool blues that contrast sharply with the warm highlights on the asphalt. The sky transitions from molten orange near the horizon to a stormy indigo overhead, with thick, billowing clouds partially obscuring the sun and casting dramatic shafts of light across the track. The image employs a hyper-realistic digital painting style with meticulous attention to textural details, from the carbon fiber weave of the car’s chassis to the individual raindrops on the track, while the color grading emphasizes high-contrast warm tones against cool shadows, lending a"

    model_type = "qwen-image-edit"
    prompt = "Change the rabbit's color to purple, with a flash light background."
    image = Image.open("pic/rabbitfeatured.jpg").convert("RGB")
    
    kwargs = {
        "max_sequence_length": max_sequence_length,
        "prompt": prompt,
        "num_images_per_prompt": num_images_per_prompt
    }
    if model_type == "qwen-image-edit":
        kwargs['image'] = image
    
    # test_free_baseline(model_type, **kwargs)
    test_learned_baseline(model_type, "checkpoints/best_val_main_loss.pt", **kwargs)



