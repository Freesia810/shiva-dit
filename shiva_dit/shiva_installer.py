import torch.nn as nn
from typing import Type, Optional, List

from shiva_dit.hook.hook_manager import HookManager
from shiva_dit.utils import global_context

from shiva_dit.modules.router import ShivaRouter
from shiva_dit.modules.gater import ShivaGaterPolicy, ShivaRatioPolicy
from shiva_dit.modules.pruner import ShivaContextPruner, ShivaPrunerExpert
from shiva_dit.modules.fuser import ShivaFuserExpert
from shiva_dit.shiva_hook import ShivaDitBlockHook


def install_shiva_hooks(
    hook_manager_cls: Type["HookManager"],
    model_type: str,
    pipe,
    model_attrs: list[str],
    hook_cls_to_install: Type[ShivaDitBlockHook],
    skip_layer_indices: Optional[List[int]] = None,
    **kwargs,
):
    shiva_mode = kwargs.pop("shiva_mode", "image")  # "image", "context", "both"

    # Image Configs
    image_ratio_mode = kwargs.pop("image_ratio_mode", "adaptive")
    fixed_image_ratio = kwargs.pop("fixed_image_ratio", 0.5)
    bottleneck_dim = kwargs.pop("bottleneck_dim", 64)
    ratio_bottleneck_dim = kwargs.pop("ratio_bottleneck_dim", 32)
    num_router_groups = kwargs.pop("num_router_groups", 4)
    layer_embed_dim = kwargs.pop("layer_embed_dim", 256)
    fuse_expert_ffn_mult = kwargs.pop("fuse_expert_ffn_mult", 2)

    # Context Configs
    context_ratio_mode = kwargs.pop("context_ratio_mode", "fixed")
    fixed_context_ratio = kwargs.pop("fixed_context_ratio", 1.0)
    context_pruner_mode = kwargs.pop("context_pruner_mode", "learned")
    context_pool_size = kwargs.pop("context_pool_size", None)
    context_proj_dim = kwargs.pop("context_proj_dim", 128)
    context_shared_config = kwargs.pop("context_shared_config", {})

    # Common Configs
    init_Ts = kwargs.pop("init_Ts", 30.0)
    init_Tk = kwargs.pop("init_Tk", 30.0)

    shared_gater_cache = kwargs.pop("shared_gater_cache", None)
    shared_ratio_cache = kwargs.pop("shared_ratio_cache", None)
    shared_context_ratio_cache = kwargs.pop("shared_context_ratio_cache", None)

    model_dim = global_context.hidden_dim
    prompt_embed_dim = global_context.encoder_hidden_dim
    if model_dim is None:
        raise ValueError("[ShivaInstaller] 'hidden_dim' not found in global_context.")

    time_embed_dim = model_dim

    all_blocks = []
    for model_attr in model_attrs:
        m = getattr(pipe, model_attr)
        for name, module in m.named_modules():
            if isinstance(module, hook_manager_cls.get_block_type(model_type)):
                full_name = f"{model_attr}.{name}"
                all_blocks.append((full_name, module))

    num_layers = len(all_blocks)
    if num_layers == 0:
        print("[ShivaInstaller] Warning: Found 0 matching blocks.")
        return

    try:
        first_block_param = next(all_blocks[0][1].parameters())
        device = first_block_param.device
        dtype = first_block_param.dtype
    except StopIteration:
        raise RuntimeError("Cannot infer device/dtype: First block has no parameters.")

    print(f"[ShivaInstaller] Found {num_layers} blocks. Mode: {shiva_mode}")

    # --- 1. Image Pruning Modules ---
    router_layer_embedder = None
    shared_gater_policy = None
    shared_image_ratio_policy = None
    shared_pruner_expert = None
    shared_fuse_expert = None
    grouped_routers = None

    if shiva_mode in ["image", "both"]:
        print("  > Instantiating Image Pruning Modules...")

        router_layer_embedder = nn.Embedding(num_layers, layer_embed_dim).to(device, dtype)

        shared_gater_policy = ShivaGaterPolicy(
            time_embed_dim=time_embed_dim, 
            layer_embed_dim=layer_embed_dim, 
            bottleneck_dim=bottleneck_dim,
            num_layers=num_layers
        ).to(device, dtype)

        if image_ratio_mode == "adaptive":
            shared_image_ratio_policy = ShivaRatioPolicy(
                total_layers=num_layers, ratio_bottleneck_dim=ratio_bottleneck_dim, init_ratio=fixed_image_ratio, time_embed_dim=time_embed_dim
            ).to(device, dtype)

        shared_pruner_expert = ShivaPrunerExpert(init_Ts=init_Ts, init_Tk=init_Tk).to(device, dtype)
        shared_fuse_expert = ShivaFuserExpert(
            hidden_dim=model_dim, ffn_mult=fuse_expert_ffn_mult
        ).to(device, dtype)

        grouped_routers = nn.ModuleList([
            ShivaRouter(
                hidden_dim=model_dim,
                time_embed_dim=time_embed_dim,
                prompt_embed_dim=prompt_embed_dim,
                layer_embed_dim=layer_embed_dim,
                bottleneck_dim=bottleneck_dim,
            )
            for _ in range(num_router_groups)
        ]).to(device, dtype)

    # --- 2. Context Pruning Modules ---
    shared_context_ratio_policy = None

    if shiva_mode in ["context", "both"]:
        print("  > Instantiating Context Modules...")

        if context_ratio_mode == "adaptive":
            shared_context_ratio_policy = ShivaRatioPolicy(
                total_layers=num_layers, ratio_bottleneck_dim=ratio_bottleneck_dim, init_ratio=fixed_context_ratio
            ).to(device, dtype)
            assert False, "Adaptive context ratio is not implemented yet."
        elif context_ratio_mode == 'linear' or context_ratio_mode == 'cosine':
            if context_pruner_mode == 'free':
                spatial_step_thresh = spatial_group_size = None
                temporal_layer_thresh = temporal_interval = None
                if (
                    "spatial_step_thresh" in context_shared_config
                    or "spatial_group_size" in context_shared_config
                ):
                    spatial_step_thresh = context_shared_config["spatial_step_thresh"]
                    spatial_group_size = context_shared_config["spatial_group_size"]
                    assert (
                        0 < spatial_step_thresh <= 1.0
                    ), "'spatial_step_thresh' must be in (0, 1]."

                if (
                    "temporal_layer_thresh" in context_shared_config
                    or "temporal_interval" in context_shared_config
                ):
                    temporal_layer_thresh = context_shared_config["temporal_layer_thresh"]
                    temporal_interval = context_shared_config["temporal_interval"]
                    if temporal_layer_thresh <= 1.0:
                        temporal_layer_thresh = int(temporal_layer_thresh * num_layers)
                    assert (
                        0 < temporal_layer_thresh <= num_layers
                    ), f"'temporal_layer_thresh' must be in (0, {num_layers}]."

                if spatial_step_thresh and spatial_group_size:
                    print(f"    - Spatial Linear Scheduling: {spatial_step_thresh} steps with group size {spatial_group_size}")
                    HookManager.set_global(
                        "shiva_context_shared_spatial",
                        {
                            "spatial_step_thresh": spatial_step_thresh,
                            "spatial_group_size": spatial_group_size,
                        },
                    )
                if temporal_layer_thresh and temporal_interval:
                    print(f"    - Temporal Linear Scheduling: {temporal_layer_thresh} layers with interval {temporal_interval}")
                    HookManager.set_global(
                        "shiva_context_shared_temporal",
                        {
                            "temporal_layer_thresh": temporal_layer_thresh,
                            "temporal_interval": temporal_interval,
                        },
                    )

            print(f"  > Instantiating Linear Schedule Context Pruning Modules... {fixed_context_ratio[0]} -> {fixed_context_ratio[1]}")

    layers_per_group = num_layers // num_router_groups if num_router_groups > 0 else num_layers

    if skip_layer_indices is None:
        skip_layer_indices = []

    injected = 0
    skipped = 0

    for i, (hook_name, module) in enumerate(all_blocks):
        if i in skip_layer_indices:
            skipped += 1
            continue

        # Image Router
        injected_router = None
        if grouped_routers is not None:
            router_group_id = min(i // layers_per_group, num_router_groups - 1)
            injected_router = grouped_routers[router_group_id]

        # Context Router, per layer
        injected_context_router = None
        if shiva_mode in ["context", "both"]:
            injected_context_router = ShivaContextPruner(
                mode=context_pruner_mode,
                pool_size=context_pool_size,
                img_dim=model_dim,
                txt_dim=prompt_embed_dim,
                proj_dim=context_proj_dim,
                init_Ts=init_Ts,
                init_Tk=init_Tk,
            ).to(device, dtype)

        hook_kwargs = {
            "shiva_mode": shiva_mode,
            "total_layers": num_layers,
            # Image Modules & Embeddings
            "router_layer_embedder": router_layer_embedder,
            "shared_router": injected_router,
            "shared_gater_policy": shared_gater_policy,
            "shared_image_ratio_policy": shared_image_ratio_policy,
            "shared_pruner": shared_pruner_expert,
            "shared_fuse_expert": shared_fuse_expert,
            "fixed_image_ratio": fixed_image_ratio,
            # Context Modules
            "shared_context_pruner": injected_context_router,
            "shared_context_ratio_policy": shared_context_ratio_policy,
            "fixed_context_ratio": fixed_context_ratio,
            "context_ratio_mode": context_ratio_mode,
            # LuT
            "shared_gater_cache": shared_gater_cache,
            "shared_ratio_cache": shared_ratio_cache,
            "shared_context_ratio_cache": shared_context_ratio_cache,
            "total_layers": num_layers,
            **kwargs,
        }

        hook = hook_cls_to_install(**hook_kwargs)
        hook.register_name(model_type, hook_name)
        if hasattr(module, "register_hook"):
            module.register_hook(hook_name, hook)
        else:
            raise RuntimeError(f"Module '{hook_name}' does not support hook registration.")

        hook_manager_cls._hook_map[hook_name] = hook
        hook_manager_cls._weak_blocks[hook_name] = module
        injected += 1

        if context_pruner_mode == "free":
            hook.eval()

    print(f"[ShivaInstaller] Injected {injected} shared hooks.")

if __name__ == "__main__":
    from shiva_dit.utils import get_pipeline, process_quick_start
    from PIL import Image

    max_sequence_length = None
    image = None
    input_images = None

    # model_type = "omnigen"
    # prompt = "A man in a black shirt is reading a book. The man is the left man in <img><|image_1|></img>."
    # input_images = ["./pic/two_man.jpg"]

    model_type = "qwen-image-edit"
    prompt = "Change the rabbit's color to purple, with a flash light background."
    image = Image.open("pic/rabbitfeatured.jpg").convert("RGB")

    # model_type = "qwen-image"
    # prompt = """### 杭州奥体中心莲花体育馆及周边建筑环境宣传海报提示词（1500字精编版）  \n\n杭州奥体中心莲花体育馆（“大莲花”）作为2022年亚运会主体育场，是杭州“山水城市”与“数字未来”的象征性地标。其设计灵感源于西湖莲花，28片钢结构花瓣环绕钱塘江岸，与“小莲花”网球中心、杭州之门双塔、城市阳台公园等共同构成“一江两岸”生态文化走廊。为打造高传播力的宣传海报，以下提示词聚焦**视觉真实性、情感共鸣与文化深度**，严格遵循杭州官方定位（避免浮夸，突出“节俭亚运”与可持续理念），并适配AI生成工具（如MidJourney）及文案设计。全文1500字，分四部分精准落地。\n\n---\n\n### **一、视觉描述类提示词（500字）——构建沉浸式场景**  \n海报需通过**高精度细节**传递场馆与自然的共生关系。提示词应明确时间、视角、元素比例，避免失真（如莲花瓣数必须为28片，周边建筑需按真实比例）。推荐使用参数：`--ar 16:9`（宽屏）、`realistic, architectural photography`（写实风格）。\n\n- **白天生态场景（主推旅游/健康主题）**：  \n  `杭州奥体中心莲花体育馆航拍，春日晴空，28片银白色钢结构花瓣在阳光下舒展，细节清晰可见；钱塘江波光粼粼作为前景，江面倒映莲花轮廓；背景钱江世纪城天际线（左侧“小莲花”网球中心、右侧杭州国际博览中心“月亮船”）、远处钱塘江大桥若隐若现；中景城市阳台公园绿意盎然，樱花树飘落花瓣，慢行步道上有晨跑市民与骑行家庭；广角镜头捕捉亚运公园湿地生态，白鹭掠过水面；风格：自然光摄影，青瓷蓝+生态绿主色调，突出“绿色亚运”遗产。--ar 16:9 --no crowds --v 6.0`  \n  *关键点：必须含钱塘江、28瓣莲花、城市阳台；避免西湖元素喧宾夺主（仅可远处加“三潭印月”剪影）。*\n\n- **夜晚活力场景（主推赛事/城市形象）**：  \n  `杭州奥体中心莲花体育馆夜景，深蓝色夜空下，花瓣LED灯光（青瓷色）勾勒动态轮廓，倒影于钱塘江水面；背景杭州之门双塔楼与钱江新城“日月同辉”灯光秀交相辉映，城市阳台观景台人流温馨；空中无人机编队组成“杭州欢迎您”汉字及亚运五色；长曝光摄影捕捉江畔车流光轨，点缀星光；风格：赛博朋克与水墨交融，避免过度炫光，强调“科技节俭”（如光伏屋顶微光）。--ar 21:9 --style raw`  \n  *关键点：灯光色系用杭州青瓷蓝（非金色），江面倒影需对称；禁用“烟花”（亚运承诺零烟花）。*\n\n- **人文互动场景（主推社区/亲子主题）**：  \n  `莲花体育馆环馆绿道特写，中景年轻父母推婴儿车漫步，背景28瓣钢结构清晰可见；钱塘江岸樱花树飘落，前景亚运吉祥物“江南忆”立牌与市民合影；远处小莲花网球中心玻璃幕墙反射江景，杭州之门双塔作远景；广角镜头纳入城市阳台草坪野餐家庭，吴语标语“灵光嘞！”手写牌；风格：温暖日光，生活化构图，突出“场馆免费开放日”亲和力。--ar 3:2 --no commercial`  \n  *关键点：人物占比≤30%，避免拥挤；必须关联“小莲花”与钱塘江。*\n\n**避坑指南**：  \n- 必含元素：钱塘江（非西湖）、28瓣莲花、城市阳台、杭州之门。  \n- 禁用元素：纯西方建筑风格（加`China architecture`参数）、奢华装饰（亚运强调节俭）、孤立场馆（需体现“一江两岸”联动）。  \n- AI优化：生成后检查花瓣数、建筑比例（小莲花高度应为大莲花1/3）。\n\n---\n\n### **二、情感氛围类提示词（300字）——锚定城市精神内核**  \n海报情感需平衡**东方美学与现代活力**，避免空洞口号。核心公式：`自然生态 × 体育精神 × 杭州基因`。\n\n- **核心氛围关键词**：  \n  `莲动钱江（动态感） | 青瓷呼吸（生态隐喻） | 赛事永续（亚运遗产） | 江畔烟火（生活温度）`  \n  *应用示例：主色调用“青瓷蓝+莲花粉”，底纹融入丝绸纹理；光影设计参考南宋水墨“留白”，避免饱和度过高。*\n\n- **推荐短语（用于设计Brief）**：  \n  - `绽放非盛放，莲瓣舒展即未来`（强调可持续：花瓣开合调节自然通风）  \n  - `钱江心跳，每一步都是亚运足迹`（关联绿道运动场景）  \n  - `一江两岸，莲语世界`（文化深度：莲花象征“出淤泥不染”，呼应杭州清廉文化）  \n  - `绿色脉搏，城市会呼吸`（突出光伏屋顶、雨水回收系统）\n\n- **雷区规避**：  \n  - 禁用“奢华”“顶级”——亚运投入仅悉尼1/3，应强调“节俭创新”（如文案加“每度电来自江风”）。  \n  - 避免孤立场馆——必须体现“莲花与钱江共生”（例如：江面上的倒影与岸边的景观融为一体）。\n\n---\n\n### **三、文化深度类提示词（300字）——彰显杭州文化底蕴**  \n海报需通过文化符号与历史底蕴，传达杭州独特的文化魅力。核心公式：`历史传承 × 文化融合 × 亚运精神`。\n\n- **核心文化关键词**：  \n  `古韵新声（古今交融） | 莲花禅意（文化象征） | 数字未来（科技引领） | 诗意栖居（生活美学）`  \n  *应用示例：主色调用“青瓷蓝+莲花粉”，底纹融入丝绸纹理；光影设计参考南宋水墨“留白”，避免饱和度过高。*\n\n- **推荐短语（用于设计Brief）**：  \n  - `古韵新声，莲花绽放新时代`（结合传统文化与现代设计）  \n  - `禅意莲心，静谧中的活力`（突出莲花的文化象征意义）  \n  - `数字未来，科技赋能亚运`（展示科技与亚运的结合）  \n  - `诗意栖居，生活即是风景`（强调杭州的生活美学）\n\n- **雷区规避**：  \n  - 避免使用过于现代或西方的元素，保持中国古典风格。  \n  - 避免过度商业化，保持文化纯粹性。  \n  - 禁用“奢华”“顶级”等词汇，强调“节俭创新”。\n\n---\n\n### **四、综合提示词（400字）——全面覆盖视觉、情感与文化**  \n综合前三部分的提示词，确保海报在视觉、情感和文化上达到最佳效果。\n\n- **综合提示词**：  \n  `杭州奥体中心莲花体育馆全景，春日晴空下，28片银白色钢结构花瓣在阳光下舒展，细节清晰可见；钱塘江波光粼粼作为前景，江面倒映莲花轮廓；背景钱江世纪城天际线（左侧“小莲花”网球中心、右侧杭州国际博览中心“月亮船”）、远处钱塘江大桥若隐若现；中景城市阳台公园绿意盎然，樱花树飘落花瓣，慢行步道上有晨跑市民与骑行家庭；广角镜头捕捉亚运公园湿地生态，白鹭掠过水面；夜晚，深蓝色夜空下，花瓣LED灯光（青瓷色）勾勒动态轮廓，倒影于钱塘江水面；背景杭州之门双塔楼与钱江新城“日月同辉”灯光秀交相辉映，城市阳台观景台人流温馨；空中无人机编队组成“杭州欢迎您”汉字及亚运五色；长曝光摄影捕捉江畔车流光轨，点缀星光；风格：自然光摄影与赛博朋克交融，避免过度炫光，强调“科技节俭”。--ar 16:9 --v 6.0`\n\n- **情感与文化提示**：  \n  `海报需传达出“莲动钱江”、“青瓷呼吸”、“赛事永续”、“江畔烟火”的情感氛围，同时结合“古韵新声”、“莲花禅意”、“数字未来”、“诗意栖居”的文化深度，确保整体画面既有现代活力又不失文化底蕴。`\n\n- **雷区规避**：  \n  - 必须包含钱塘江、28瓣莲花、城市阳台、杭州之门等关键元素。  \n  - 禁用“奢华”“顶级”等词汇，强调“节俭创新”。  \n  - 避免纯西方建筑风格，保持中国古典风格。  \n  - 生成后检查花瓣数、建筑比例，确保画面真实可信。,"""
    # max_sequence_length = 4096

    pipe, model_attrs = get_pipeline(model_type)

    shiva_config = {
        "shiva_mode": "context",
        "context_ratio_mode": "cosine",
        "fixed_context_ratio": (1.0, 0.1),
        "context_pruner_mode": "free",
        "context_pool_size": 64,
        "context_shared_config": {
            "spatial_step_thresh": 0.2,
            "spatial_group_size": 2,
            "temporal_layer_thresh": 60,
            "temporal_interval": 4,
        },
        "init_Ts": 10.0,
        "init_Tk": 10.0,
    }

    HookManager.configure_hooks(
        model_type=model_type,
        default=(ShivaDitBlockHook, shiva_config),
    )

    with HookManager.enabled():
        HookManager.inject_hooks(
            model_type=model_type, pipe=pipe, model_attrs=model_attrs
        )
        if model_type == "qwen-image-edit":
            res = process_quick_start(pipe, prompt=prompt, max_sequence_length=max_sequence_length, image=image, num_images_per_prompt=1)
        elif model_type == "omnigen":
            res = process_quick_start(pipe, prompt=prompt, input_images=input_images, num_images_per_prompt=1)
        else:
            res = process_quick_start(pipe, prompt=prompt, max_sequence_length=max_sequence_length, num_images_per_prompt=1)
        res.save(f"{model_type}_shiva.png")

    print(global_context)
