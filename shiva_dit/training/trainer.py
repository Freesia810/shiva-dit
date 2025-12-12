import os
import json
import copy
import re
import glob
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import ModuleType
from torch.utils.data import DataLoader
from peft import LoraConfig as PeftLoraConfig, get_peft_model
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs, broadcast_object_list
from dataclasses import asdict

try:
    from torchmetrics.multimodal.clip_score import CLIPScore
    from torchmetrics.image.kid import KernelInceptionDistance
    from torchmetrics.multimodal import CLIPImageQualityAssessment
    import torchvision.transforms.functional as TF
except ImportError:
    CLIPScore = None
    KernelInceptionDistance = None
    CLIPImageQualityAssessment = None
    print("Warning: torchmetrics not installed. Metrics (CLIP Score, KID, IQA) will not be calculated.")

LOCAL_BERT_PATH = "../bert-base-uncased"

if not os.path.exists(LOCAL_BERT_PATH):
    print(f"Warning: Local BERT path not found at {LOCAL_BERT_PATH}. Offline mode might fail.")

from transformers import BertTokenizer, BertModel

_original_tokenizer_from_pretrained = BertTokenizer.from_pretrained
_original_model_from_pretrained = BertModel.from_pretrained

def _patched_tokenizer_loader(cls, pretrained_model_name_or_path, *args, **kwargs):
    if pretrained_model_name_or_path == "bert-base-uncased":
        if os.path.exists(LOCAL_BERT_PATH):
            print(f"Intercepted 'bert-base-uncased' -> Redirecting to local: {LOCAL_BERT_PATH}")
            pretrained_model_name_or_path = LOCAL_BERT_PATH
    return _original_tokenizer_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

def _patched_model_loader(cls, pretrained_model_name_or_path, *args, **kwargs):
    if pretrained_model_name_or_path == "bert-base-uncased":
        if os.path.exists(LOCAL_BERT_PATH):
            print(f"Intercepted 'bert-base-uncased' (Model) -> Redirecting to local: {LOCAL_BERT_PATH}")
            pretrained_model_name_or_path = LOCAL_BERT_PATH
    return _original_model_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

BertTokenizer.from_pretrained = classmethod(_patched_tokenizer_loader)
BertModel.from_pretrained = classmethod(_patched_model_loader)

sys.modules["ImageReward.ReFL"] = ModuleType("ImageReward.ReFL")
import transformers.modeling_utils
import transformers.pytorch_utils
if not hasattr(transformers.modeling_utils, "apply_chunking_to_forward"):
    if hasattr(transformers.pytorch_utils, "apply_chunking_to_forward"):
        transformers.modeling_utils.apply_chunking_to_forward = transformers.pytorch_utils.apply_chunking_to_forward

if not hasattr(transformers.modeling_utils, "find_pruneable_heads_and_indices"):
    if hasattr(transformers.pytorch_utils, "find_pruneable_heads_and_indices"):
        transformers.modeling_utils.find_pruneable_heads_and_indices = transformers.pytorch_utils.find_pruneable_heads_and_indices

if not hasattr(transformers.modeling_utils, "prune_linear_layer"):
    if hasattr(transformers.pytorch_utils, "prune_linear_layer"):
        transformers.modeling_utils.prune_linear_layer = transformers.pytorch_utils.prune_linear_layer

try:
    import ImageReward as RM
except ImportError:
    RM = None
    print("Warning: ImageReward not installed. ImageReward Score will not be calculated.")

from baselines.diffcr import DiffCRDitBlockHook
from baselines.dy_dit import DynamicDiTBlockHook
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from shiva_dit.hook import HookManager
from shiva_dit.utils import get_pipeline
from shiva_dit.training.input_preparers import get_input_preparer
from shiva_dit.training.config import TrainerConfig
from shiva_dit.training.dataset import ImageTextDataset, collate_fn
from shiva_dit.training.logging import NoOpLogger
from shiva_dit.training.metric_manager import MetricManager
from shiva_dit.utils import global_context
import shiva_dit.shiva_metrics as shiva_metrics
try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    load_safetensors = None


class DiffusionModelTrainer:
    def __init__(self, config: TrainerConfig, load_evaluators: bool = True):
        self.config = config
        self.model_type = config.model_type
        self.current_stage_idx = -1

        self.input_preparer_fn = get_input_preparer(self.model_type)

        ## Initialize accelerator, device, seed and generator
        ddp_kwargs = DistributedDataParallelKwargs(static_graph=True)
        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="tensorboard" if config.log_with_tensorboard else None,
            project_dir=os.path.join(config.output_dir, config.tensorboard_log_dir),
            kwargs_handlers=[ddp_kwargs],
        )
        self.device = self.accelerator.device
        set_seed(config.seed)
        self.generator = torch.Generator(device=self.accelerator.device).manual_seed(
            config.seed
        )

        self._setup_models_and_components()

        self._setup_hooks()

        self.skip_layer_indices = []
        if self.config.trainable_hook_config:
            _, hook_kwargs = self.config.trainable_hook_config
            self.skip_layer_indices = hook_kwargs.get("skip_layer_indices", [])
            if self.accelerator.is_main_process:
                print(f"[Trainer] Skip layer indices: {self.skip_layer_indices}")

        self.optimizer = self._setup_model_and_optimizer()

        self._setup_dataloaders()

        num_update_steps_per_epoch = max(
            1, len(self.train_dataloader) // config.gradient_accumulation_steps
        )
        self.num_training_steps = num_update_steps_per_epoch * config.num_epochs

        self._setup_scaled_steps()

        self._prepare_with_accelerator()

        self.logger = self._setup_logging()

        self._setup_metric_manager()

        self._setup_caption_dropout_embeds()

        self.best_metrics_state = {}

        if self.accelerator.is_main_process and load_evaluators:
            self.evaluators = {}
            if CLIPScore:
                print("Loading CLIP Score model...")
                self.evaluators['clip'] = CLIPScore(model_name_or_path="../openai-clip-vit-base-patch16", sync_on_compute=False).to(self.device)
            if RM:
                print("Loading ImageReward model...")
                self.evaluators['image_reward'] = RM.load("../ImageReward/ImageReward.pt", device=self.device, med_config="../ImageReward/med_config.json")

            if KernelInceptionDistance:
                print("Loading KID model...")
                self.evaluators['kid'] = KernelInceptionDistance(
                    subset_size=10, reset_real_features=False, sync_on_compute=False
                ).to(self.device)
                self.kid_stats_ready = False
            if CLIPImageQualityAssessment:
                print("Loading CLIP IQA model...")
                self.evaluators['clip_iqa'] = CLIPImageQualityAssessment(model_name_or_path="../openai-clip-vit-base-patch16", sync_on_compute=False).to(self.device)

        self._print_setup_summary()

        self.main_viz_buffer = []
        self.distill_final_viz_buffer = []
        self.distill_mid_viz_buffer = []

        self.state_buffer = {}
        local_seed = self.config.seed + self.accelerator.process_index
        self.local_generator = torch.Generator(device=self.device).manual_seed(local_seed)
        print(f"[Trainer] Device [{self.device}] Set Local seed: {local_seed}")

        self.timestep_buffer = []
        self.schedule_timesteps_gpu = self.scheduler.timesteps.to(self.device)

    def _scale_steps(self, steps: int) -> int:
        if steps <= 0:
            return steps
        return max(1, steps // self.accelerator.num_processes)

    def _setup_scaled_steps(self):
        config = self.config

        total_train_steps = self.num_training_steps

        self.processed_stage_schedule = []
        current_step_threshold = 0
        last_threshold = 0 

        if 'stage_schedule' not in self.config.extra_config:
            self.processed_stage_schedule.append({
                "threshold": self._scale_steps(total_train_steps),
                "start_step": 0,
                "weights": {"ratio": 0.5, "gate": 0.1},
                "lr": None,
                "temp": [5.0, 5.0],
            })
        else:
            schedule_list = self.config.extra_config['stage_schedule']
            for i, stage in enumerate(schedule_list):
                stage_duration = stage["steps"]
                start_step = current_step_threshold
                is_last_stage = (i == len(schedule_list) - 1)
                scaled_total_steps = self._scale_steps(total_train_steps)
                if stage_duration < 0 or is_last_stage:
                    scaled_threshold = scaled_total_steps
                else:
                    scaled_duration = self._scale_steps(stage_duration)
                    scaled_threshold = current_step_threshold + scaled_duration
                    if scaled_threshold > scaled_total_steps:
                        scaled_threshold = scaled_total_steps

                self.processed_stage_schedule.append({
                    "threshold": scaled_threshold,
                    "start_step": int(start_step),
                    "weights": stage["weights"],
                    "lr": stage.get("lr", None),
                    "temp": stage.get("temp", [5.0, 5.0]),
                })

                current_step_threshold = scaled_threshold
                last_threshold = scaled_threshold
                if current_step_threshold >= total_train_steps:
                    break

        self.scaled_hook_only_steps = self._scale_steps(config.hook_only_steps)
        self.log_trigger = self._scale_steps(config.logging_steps)
        self.validate_trigger = (
            self._scale_steps(config.validation_steps)
            if config.validation_steps > 0
            else None
        )
        self.inference_trigger = (
            self._scale_steps(config.inference_steps)
            if config.inference_steps > 0
            else None
        )
        self.save_trigger = (
            self._scale_steps(config.save_steps) if config.save_steps > 0 else None
        )

    @staticmethod
    def _get_dtype(mixed_precision: str):
        if mixed_precision == "bf16":
            dtype = torch.bfloat16
        elif mixed_precision == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        return dtype

    def _setup_models_and_components(self):
        config = self.config
        # get pipeline and components
        self.pipe, self.model_attrs = get_pipeline(
            self.model_type,
            model_path=config.model_path,
            dtype=DiffusionModelTrainer._get_dtype(config.mixed_precision),
        )

        self.image_processor = self.pipe.image_processor
        self.vae = self.pipe.vae
        self.scheduler = self.pipe.scheduler
        self.model = tuple(
            getattr(self.pipe, model) for model in self.model_attrs
        )  # DiT
        assert len(self.model) == 1, "Currently only single DiT model is supported."
        self.model = self.model[0]

        # distillation setup
        self.teacher_model = None
        if self.config.distillation_final_weight > 0 or self.config.distillation_middle_weight > 0:
            if self.accelerator.is_main_process:
                print("Distillation loss enabled. Loading static teacher model...")

            teacher_pipe, teacher_attrs = get_pipeline(
                self.model_type,
                model_path=config.model_path,
                dtype=DiffusionModelTrainer._get_dtype(config.mixed_precision),
            )
            self.teacher_model = getattr(teacher_pipe, teacher_attrs[0])
            assert len(teacher_attrs) == 1, "Currently only single DiT model is supported."
            self.teacher_model.requires_grad_(False)
            self.teacher_model.eval()
            self.teacher_model = self.teacher_model.to(self.device)
            del teacher_pipe # only need DiT model
            if self.accelerator.is_main_process:
                print("Teacher model loaded.")

        # get tokenizers and text encoders
        if self.model_type in ["flux-dev", "flux-kontext"]:
            self.tokenizers = (self.pipe.tokenizer, self.pipe.tokenizer_2)
            self.text_encoders = (self.pipe.text_encoder, self.pipe.text_encoder_2)
            self.text_encoder_names = ("text_encoder", "text_encoder_2")
        elif self.model_type in ["pixart-sigma", "qwen-image", "qwen-image-edit", "wan"]:
            self.tokenizers = (self.pipe.tokenizer,)
            self.text_encoders = (self.pipe.text_encoder,)
            self.text_encoder_names = ("text_encoder",)
        elif self.model_type == "sd3":
            self.tokenizers = (
                self.pipe.tokenizer,
                self.pipe.tokenizer_2,
                self.pipe.tokenizer_3,
            )
            self.text_encoders = (
                self.pipe.text_encoder,
                self.pipe.text_encoder_2,
                self.pipe.text_encoder_3,
            )
            self.text_encoder_names = (
                "text_encoder",
                "text_encoder_2",
                "text_encoder_3",
            )
        else:
            raise ValueError(f"Undefined model_type: {self.model_type}")

        if self.image_processor is None or self.vae is None or self.model is None:
            raise ValueError("NoneType found in image_processor, vae or model.")

        # freeze vae and text encoder(s)
        self.vae.requires_grad_(False)
        self.vae.eval()
        for text_encoder in self.text_encoders:
            text_encoder.requires_grad_(False)
            text_encoder.to(dtype=torch.float32) # keep text encoder in fp32 for stability
            text_encoder.eval()

    def _setup_hooks(self):
        config = self.config
        self.hook_type = None
        if config.trainable_hook_config:
            hook_class, hook_kwargs = config.trainable_hook_config

            if hook_class.__name__ == "DynamicDiTBlockHook":
                self.hook_type = "dydit"
            elif hook_class.__name__ == "DiffCRDitBlockHook":
                self.hook_type = "diffcr"
            elif hook_class.__name__ == "ShivaDitBlockHook":
                self.hook_type = "shiva"
                hook_kwargs["capture_mask_for_viz"] = True
                hook_kwargs["skip_layer_indices"] = [0]
                if self.accelerator.is_main_process:
                    print("Hook type: Shiva. Will use shared injection via setup_shared_hooks.")
            else:
                raise ValueError(f"Unknown hook class: {hook_class.__name__}")

            HookManager.configure_hooks(
                model_type=config.model_type,
                default=(hook_class, hook_kwargs),
            )
            HookManager.enable()
            HookManager.inject_hooks(
                model_type=config.model_type,
                pipe=self.pipe,
                model_attrs=self.model_attrs,
            )

    def _setup_dataloaders(self):
        config = self.config
        self.train_dataset = ImageTextDataset(
            model_type=self.model_type,
            metadata_file=config.metadata_file,
            image_dir=config.image_dir,
            image_processor=self.pipe.image_processor,
            tokenizers=self.tokenizers,
            pipe=self.pipe,
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=config.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        self.val_dataloader = None
        if config.val_metadata_file and config.val_image_dir:
            try:
                val_dataset = ImageTextDataset(
                    model_type=self.model_type,
                    metadata_file=config.val_metadata_file,
                    image_dir=config.val_image_dir,
                    image_processor=self.pipe.image_processor,
                    tokenizers=self.tokenizers,
                    pipe=self.pipe,
                )
                val_batch_size = (
                    config.val_batch_size
                    if config.val_batch_size
                    else config.train_batch_size
                )
                self.val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=val_batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                    pin_memory=True,
                )
            except Exception as e:
                print(
                    f"Warning: Failed to load validation dataset. Validation will be skipped. Error: {e}"
                )
                self.val_dataloader = None

    def _prepare_with_accelerator(self):
        config = self.config
        prepare_list = [
            self.model,
            self.optimizer,
            self.train_dataloader,
        ]

        # if self.teacher_model is not None:
        #     prepare_list.insert(1, self.teacher_model)

        prepared = self.accelerator.prepare(*prepare_list)

        idx = 0
        self.model = prepared[idx]; idx += 1
        # if self.teacher_model is not None:
        #     self.teacher_model = prepared[idx]; idx += 1
        self.optimizer = prepared[idx]; idx += 1
        self.train_dataloader = prepared[idx]; idx += 1
        if self.val_dataloader is not None:
            self.val_dataloader = self.accelerator.prepare(self.val_dataloader)

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        self.dtype = unwrapped_model.dtype # final model dtype after accelerator prepare

        if self.accelerator.is_main_process:
            print("model_type:", config.model_type)
            print("image_processor:", self.image_processor.__class__.__name__)
            print("scheduler:", self.scheduler.__class__.__name__)
            print("vae:", self.vae.__class__.__name__, "dtype:", self.vae.dtype)
            print("text_encoder:", self.text_encoders[0].__class__.__name__, "dtype:", self.text_encoders[0].dtype)
            print("model (DiT):", self.model.__class__.__name__, "dtype:", self.dtype)

    def _setup_logging(self):
        if not self.accelerator.is_main_process:
            return NoOpLogger()
        config = self.config
        try:
            from shiva_dit.training.logging import (
                TensorBoardLogger, WandbLogger, MLflowLogger, LoggerCollection
            )
        except ImportError as e:
            print(f"Not able to import logging modules: {e}. Logging will be disabled.")
            return NoOpLogger()

        loggers = []

        if config.log_with_tensorboard:
            try:
                tb_log_path = os.path.join(config.output_dir, config.tensorboard_log_dir)
                loggers.append(TensorBoardLogger(log_dir=tb_log_path))
            except ImportError:
                print("Warning: Unable to import TensorBoard, skipping TensorBoard logging.")

        if getattr(config, "log_with_wandb", False):
            try:
                loggers.append(
                    WandbLogger(
                        project=getattr(config, "wandb_project", "diffusion_trainer"),
                        name=getattr(
                            config,
                            "wandb_run_name",
                            os.path.basename(config.output_dir),
                        ),
                        config_dict=asdict(config),
                        entity=getattr(config, "wandb_entity", None),
                    )
                )
            except ImportError:
                print("Warning: Unable to import wandb, skipping Wandb logging.")

        if getattr(config, "log_with_mlflow", False):
            try:
                run_name = config.mlflow_run_name or config.wandb_run_name or os.path.basename(config.output_dir)

                loggers.append(MLflowLogger(
                    experiment_name=config.mlflow_experiment_name,
                    run_name=run_name,
                    config_dict=asdict(config),
                    tracking_uri=config.mlflow_tracking_uri
                ))
            except ImportError:
                print("Warning: Unable to import mlflow, skipping MLflow logging.")

        if not loggers:
            print("No logging backends available, logging will be disabled.")
            return NoOpLogger()

        return LoggerCollection(loggers)

    def _setup_metric_manager(self):
        self.metric_grouping_rules = {
            "loss/losses_weighted": [
                "loss/image_ratio/ratio_loss",
                "loss/context_ratio/ratio_loss", 
                "loss/context/context_aux_loss",
                "loss/gater/gate_loss",
                "loss/diffcr_loss",
            ],
            "loss/losses_raw": [
                "loss/gater/fuse_penalty",
                "loss/image_ratio/floor_loss",
                "loss/image_ratio/mse_loss",
                "loss/image_ratio/budget_loss",
                "loss/context_ratio/floor_loss",
                "loss/context_ratio/mse_loss",
                "loss/context_ratio/budget_loss",
                "loss/context/context_aux_loss_raw",
            ],
            "router/scores": [
                "router/mean_score",
                "router/min_score",
                "router/max_score",
                "router/p25_score",
                "router/p50_score",
                "router/p75_score",
            ],
            "router/distribution": [
                "router/std_score",
            ],
            "ratio_policy/image_ratio": [
                "image_ratio/target_ratio",
                "image_ratio/avg_ratio",
                "image_ratio/cycle_mean_est",
            ],
            "ratio_policy/distribution": [
                "image_ratio/std_ratio",
                "context_ratio/std_ratio",
            ],
            "ratio_policy/context_ratio": [ 
                "context_ratio/avg_ratio",
                "context_ratio/target_ratio",
            ],
            "gater": [
                "gater/avg_fuse_prob",
                "gater/max_fuse_prob",
                "gater/min_fuse_prob",
            ],
            "context/scores": [
                "context/align_score_mean",
                "context/align_score_max",
                "context/align_score_min",
            ],
            "context/distribution": [
                "context/align_score_std",
            ],
            "grad/img_policy/grad_std": [
                "grad/img_policy/temb_proj_std",
                "grad/img_policy/layer_proj_std",
            ],
            "topk/status": [
                "topk/status/active_tokens",
                "topk/status/boundary_width",
                "topk/status/rank_spread",
                "topk/status/k_target"
            ],
            "topk/grad": [
                "topk/grad/logits_norm",
                "topk/grad/k_norm",
                "topk/grad/k_mean",
            ],

            "grad/module": [
                "grad/router_avg_norm",
                "grad/gater_policy_avg_norm",
                "grad/image_ratio_policy_avg_norm",
                "grad/context_ratio_policy_avg_norm",
                "grad/context_pruner_avg_norm",
            ],

            "diffcr/stats": [
                "diffcr/avg_ratio",
                "diffcr/min_ratio",
                "diffcr/max_ratio",
                "diffcr/target_ratio",
            ],
        }

        self.metric_manager = MetricManager(
            self.logger, self.accelerator, self.metric_grouping_rules
        )
        self.step_metrics = {}
        self.step_per_layer_metrics = {}

    def _setup_model_and_optimizer(self):
        # freeze VAE and Text Encoder(s)
        self.vae.requires_grad_(False)
        for text_encoder in self.text_encoders:
            text_encoder.requires_grad_(False)

        if self.config.training_mode == "hook_only":
            if self.accelerator.is_main_process:
                print("Training mode: hook_only (only training Hook parameters)")
            self.model.requires_grad_(False)

            hook_param_found = False
            for hook in HookManager.get_all_hooks().values():
                if isinstance(hook, nn.Module):
                    hook.requires_grad_(True)
                    if list(hook.parameters()):
                        hook_param_found = True

            if not hook_param_found:
                all_hook_params = []
                for hook in HookManager.get_all_hooks().values():
                    if isinstance(hook, nn.Module):
                        all_hook_params.extend(list(hook.parameters()))
                if not all_hook_params:
                    raise ValueError(
                        "No trainable Hook parameters found in 'hook_only' mode."
                    )

        elif self.config.training_mode == "finetune":
            if self.accelerator.is_main_process:
                print("Training mode: finetune (training all DiT parameters)")
            self.model.requires_grad_(True)
            for hook in HookManager.get_all_hooks().values():
                if isinstance(hook, nn.Module):
                    hook.requires_grad_(True)

        elif self.config.training_mode == "lora":
            if self.accelerator.is_main_process:
                print("Training mode: lora (applying LoRA to DiT model)")
            if not self.config.lora_config.enable:
                raise ValueError(
                    "LoRA configuration is disabled in 'lora' training mode."
                )

            self.model.requires_grad_(False)
            for hook in HookManager.get_all_hooks().values():
                if isinstance(hook, nn.Module):
                    hook.requires_grad_(False)

            lora_conf = self.config.lora_config
            peft_config = PeftLoraConfig(
                r=lora_conf.rank,
                lora_alpha=lora_conf.lora_alpha,
                target_modules=lora_conf.target_modules,
                lora_dropout=lora_conf.lora_dropout,
                bias="none",
            )
            self.model = get_peft_model(self.model, peft_config)

            if self.accelerator.is_main_process:
                self.model.print_trainable_parameters()

            for hook in HookManager.get_all_hooks().values():
                if isinstance(hook, nn.Module):
                    hook.requires_grad_(True)
        else:
            raise ValueError(f"Unknown training mode: {self.config.training_mode}")

        if self.config.enable_gradient_checkpointing:
            self.model.enable_gradient_checkpointing()
            if self.accelerator.is_main_process:
                print("Enabled gradient checkpointing for the model.")

        lora_params = []
        lora_names = []
        
        gate_policy_params = []
        gate_policy_names = []
        
        image_ratio_policy_params = []
        image_ratio_policy_names = []
        
        fuser_params = []
        fuser_names = []
        
        context_ratio_policy_params = []
        context_ratio_policy_names = []
        
        context_pruner_params = []
        context_pruner_names = []
        
        other_model_params = []
        other_model_names = []

        gate_policy_param_ids = set()
        image_ratio_policy_param_ids = set()
        fuser_param_ids = set()
        context_ratio_policy_param_ids = set()
        context_pruner_param_ids = set()

        # Identify parameters belonging to specific modules
        for hook in HookManager.get_all_hooks().values():
            if hasattr(hook, "gater_policy") and hook.gater_policy is not None:
                for param in hook.gater_policy.parameters():
                    gate_policy_param_ids.add(id(param))
            if hasattr(hook, "image_ratio_policy") and hook.image_ratio_policy is not None:
                for param in hook.image_ratio_policy.parameters():
                    image_ratio_policy_param_ids.add(id(param))
            if hasattr(hook, "fuse_expert") and hook.fuse_expert is not None:
                for param in hook.fuse_expert.parameters():
                    fuser_param_ids.add(id(param))
            if hasattr(hook, "context_ratio_policy") and hook.context_ratio_policy is not None:
                for param in hook.context_ratio_policy.parameters():
                    context_ratio_policy_param_ids.add(id(param))
            if hasattr(hook, "context_pruner") and isinstance(hook.context_pruner, nn.Module):
                for param in hook.context_pruner.parameters():
                    context_pruner_param_ids.add(id(param))

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            pid = id(param)
            if pid in gate_policy_param_ids:
                gate_policy_params.append(param)
                gate_policy_names.append(name)
            elif pid in image_ratio_policy_param_ids:
                image_ratio_policy_params.append(param)
                image_ratio_policy_names.append(name)
            elif pid in fuser_param_ids:
                fuser_params.append(param)
                fuser_names.append(name)
            elif pid in context_ratio_policy_param_ids:
                context_ratio_policy_params.append(param)
                context_ratio_policy_names.append(name)
            elif pid in context_pruner_param_ids:
                context_pruner_params.append(param)
                context_pruner_names.append(name)
            elif "lora" in name:
                lora_params.append(param)
                lora_names.append(name)
            else:
                other_model_params.append(param)
                other_model_names.append(name)

        base_lr_params = other_model_params
        base_lr_names = other_model_names

        if self.accelerator.is_main_process:
            print(f"\n--- Optimizer Parameter Groups ---")
            
            def print_group_info(group_name, params, names):
                print(f"Optimizing {len(params)} {group_name} params")
                if len(names) > 0:
                    if len(names) > 20:
                        print(f"  Names (first 10): {names[:10]}")
                        print(f"  Names (last 5):   {names[-5:]}")
                    else:
                        print(f"  Names: {names}")
                print("-" * 30)

            print_group_info("LoRA", lora_params, lora_names)
            print_group_info("Gate Policy", gate_policy_params, gate_policy_names)
            print_group_info("Image Ratio Policy", image_ratio_policy_params, image_ratio_policy_names)
            print_group_info("Fuser", fuser_params, fuser_names)
            print_group_info("Context Ratio Policy", context_ratio_policy_params, context_ratio_policy_names)
            print_group_info("Context Pruner", context_pruner_params, context_pruner_names)
            print_group_info("Base (Router/Other)", base_lr_params, base_lr_names)
            print("----------------------------------\n")

        # Define explicit groups
        optimizer_grouped_parameters = [
            {
                "params": lora_params,
                "lr": self.config.lora_learning_rate,
                "weight_decay": self.config.weight_decay,
                "name": "lora"
            },
            {
                "params": gate_policy_params,
                "lr": self.config.policy_learning_rate,
                "weight_decay": 0.0,
                "name": "gate_policy"
            },
            {
                "params": image_ratio_policy_params,
                "lr": self.config.policy_learning_rate,
                "weight_decay": 0.0,
                "name": "image_ratio_policy"
            },
            {
                "params": fuser_params,
                "lr": self.config.learning_rate,
                "weight_decay": 0.0,
                "name": "fuser"
            },
            {
                "params": context_ratio_policy_params,
                "lr": self.config.policy_learning_rate,
                "weight_decay": 0.0,
                "name": "context_ratio_policy"
            },
            {
                "params": context_pruner_params,
                "lr": self.config.context_learning_rate, 
                "weight_decay": 0.0,
                "name": "context_pruner"
            },
            {
                "params": base_lr_params,
                "lr": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "name": "base"
            },
        ]

        # Filter out empty groups
        optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if g["params"]]

        if not optimizer_grouped_parameters:
            raise ValueError("No trainable parameters found for the optimizer.")

        if self.config.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

        return optimizer

    def _setup_caption_dropout_embeds(self):
        if self.config.caption_dropout_rate > 0:
            if self.accelerator.is_main_process:
                print(
                    f"Caption dropout enabled with rate: {self.config.caption_dropout_rate}"
                )
            if self.model_type == "pixart-sigma":
                self.empty_prompt_embeds, self.empty_prompt_attention_mask, _, _ = (
                    self.pipe.encode_prompt(
                        "",
                        do_classifier_free_guidance=False,
                        device=self.device,
                        clean_caption=True,
                    )
                )
                self.empty_prompt_embeds = self.empty_prompt_embeds.to(self.dtype)
                self.empty_prompt_attention_mask = self.empty_prompt_attention_mask.to(
                    self.dtype
                )
            elif self.model_type in ["flux-dev", "flux-kontext"]:
                (
                    self.empty_prompt_embeds,
                    self.empty_pooled_prompt_embeds,
                    self.empty_text_ids,
                ) = self.pipe.encode_prompt(
                    prompt="",
                    device=self.device,
                    num_images_per_prompt=1,
                )
                self.empty_prompt_embeds = self.empty_prompt_embeds.to(self.dtype)
                self.empty_pooled_prompt_embeds = self.empty_pooled_prompt_embeds.to(
                    self.dtype
                )
                self.empty_text_ids = self.empty_text_ids.to(self.dtype)
            elif self.model_type == "sd3":
                (
                    self.empty_prompt_embeds,
                    _,
                    self.empty_pooled_prompt_embeds,
                    _,
                ) = self.pipe.encode_prompt(
                    prompt="",
                    prompt_2=None,
                    prompt_3=None,
                    device=self.device,
                    do_classifier_free_guidance=False,
                    num_images_per_prompt=1,
                )
                self.empty_prompt_embeds = self.empty_prompt_embeds.to(self.dtype)
                self.empty_pooled_prompt_embeds = self.empty_pooled_prompt_embeds.to(
                    self.dtype
                )
            else:
                raise NotImplementedError(
                    "Caption dropout is not implemented for model_type: " + self.model_type
                )

    def _print_setup_summary(self):
        if self.accelerator.is_main_process:
            config = self.config
            print(f"\n{'='*20} Training Config {'='*20}")
            print(f"  Mode:                {config.training_mode}")
            print(f"  Micro Batch Size:    {config.train_batch_size}")
            print(f"  Grad Accum Steps:    {config.gradient_accumulation_steps}")
            print(f"  Global Batch Size:   {config.train_batch_size * config.gradient_accumulation_steps * self.accelerator.num_processes}")
            print(f"  Total Steps:         {self.num_training_steps}")
            print(f"{'='*57}\n")

    @staticmethod
    def stratified_timestep_sampler(batch_size, num_timesteps, device, generator=None):
        if batch_size > num_timesteps:
            return torch.randint(0, num_timesteps, (batch_size,), device=device, generator=generator)

        boundaries = torch.linspace(0, num_timesteps, batch_size + 1, device=device).long()
        
        start_indices = boundaries[:-1]
        
        spans = boundaries[1:] - boundaries[:-1]
        
        offsets = (torch.rand(batch_size, device=device, generator=generator) * spans).long()
        
        indices = start_indices + offsets

        indices = indices[torch.randperm(batch_size, device=device, generator=generator)]
        
        return indices

    def _sample_stratified_indices(self, micro_batch_size):
        if len(self.timestep_buffer) < micro_batch_size:
            current_global_step = self.global_step
            DEBUG_INTERVAL = 50 

            should_debug_print = (
                (current_global_step % DEBUG_INTERVAL == 0)
            )
            
            accum_steps = self.config.gradient_accumulation_steps
            total_batch_size = micro_batch_size * accum_steps
            total_batch_size = max(total_batch_size, micro_batch_size)

            rng_gen = self.local_generator

            new_indices = DiffusionModelTrainer.stratified_timestep_sampler(
                total_batch_size,
                self.scheduler.config.num_train_timesteps,
                self.device,
                generator=rng_gen
            )
            
            if should_debug_print:
                all_new_indices = self.accelerator.gather(new_indices)
                if self.accelerator.is_main_process:
                    all_indices_list = all_new_indices.cpu().flatten().tolist()
                    all_indices_list.sort()
                    
                    print(f"\n{'='*20} [Global Timestep Sampling @ Step {current_global_step}] {'='*20}")
                    print(f"Total Batches Sampled: {total_batch_size * self.accelerator.num_processes}")
                    print(f"Sampled Indices (Sorted): {all_indices_list}")
                    
                    unique_indices = set(all_indices_list)
                    print(f"Unique Indices Count: {len(unique_indices)} / {len(all_indices_list)}")
                    print(f"{'='*65}\n")

                self.accelerator.wait_for_everyone()

            self.timestep_buffer.extend(new_indices.tolist())

        current_indices_list = self.timestep_buffer[:micro_batch_size]
        
        self.timestep_buffer = self.timestep_buffer[micro_batch_size:]

        return torch.tensor(current_indices_list, device=self.device, dtype=torch.long)

    def prepare_model_inputs(self, batch, timesteps=None, timestep_index=None, generator=None):
        if timesteps is None:
            if timestep_index is None:
                batch_size = batch["pixel_values"].shape[0]
                timestep_index = self._sample_stratified_indices(batch_size)
            elif not isinstance(timestep_index, torch.Tensor):
                timestep_index = torch.tensor(timestep_index, device=self.device, dtype=torch.long)
            if timestep_index.dim() == 0:
                timestep_index = timestep_index.unsqueeze(0)
            timesteps = self.schedule_timesteps_gpu[timestep_index]

        img_latents, noise, noisy_latents, timesteps, model_kwargs = (
            self.input_preparer_fn(self, batch, timesteps, generator)
        )

        # update global context
        patch_size = 2
        hidden_seq_height = noise.shape[-2] // patch_size
        hidden_seq_width = noise.shape[-1] // patch_size
        hidden_seq_len = hidden_seq_height * hidden_seq_width
        global_context.update(hidden_seq_len=hidden_seq_len, hidden_seq_width=hidden_seq_width, hidden_seq_height=hidden_seq_height)
        if "encoder_hidden_states" in model_kwargs:
            global_context.update(
                encoder_hidden_seq_len=model_kwargs["encoder_hidden_states"].size(1)
            )

        if self.model_type == "flux-dev":
            # pack for hidden state
            batch_size = batch["pixel_values"].shape[0]
            height, width = noisy_latents.shape[-2:]
            num_channels_latents = self.pipe.transformer.config.in_channels // 4
            img_latents = self.pipe._pack_latents(
                img_latents, batch_size, num_channels_latents, height, width
            )
            noisy_latents = self.pipe._pack_latents(
                noisy_latents, batch_size, num_channels_latents, height, width
            )
            noise = self.pipe._pack_latents(
                noise, batch_size, num_channels_latents, height, width
            )
            timesteps = timesteps / 1000
        elif self.model_type == 'qwen-image-edit':
            img_latents, cond_latents = img_latents

            hidden_extra_seq_height = cond_latents.shape[-2] // patch_size
            hidden_extra_seq_width = cond_latents.shape[-1] // patch_size
            hidden_extra_seq_len = hidden_extra_seq_height * hidden_extra_seq_width
            global_context.update(hidden_extra_seq_len=hidden_extra_seq_len, hidden_extra_seq_height=hidden_extra_seq_height, hidden_extra_seq_width=hidden_extra_seq_width)

            # pack for hidden state
            batch_size = batch["pixel_values"].shape[0]
            height, width = noisy_latents.shape[-2:]
            num_channels_latents = self.pipe.transformer.config.in_channels // 4
            img_latents = self.pipe._pack_latents(
                img_latents, batch_size, num_channels_latents, height, width
            )
            noisy_latents = self.pipe._pack_latents(
                noisy_latents, batch_size, num_channels_latents, height, width
            )
            noise = self.pipe._pack_latents(
                noise, batch_size, num_channels_latents, height, width
            )
            cond_latents = self.pipe._pack_latents(
                cond_latents, batch_size, num_channels_latents, height, width
            )
            timesteps = timesteps / 1000

            noisy_latents = torch.cat([noisy_latents, cond_latents], dim=1)

        if self.model_type in ["flux-dev", "qwen-image-edit"]:
            global_context.update(current_timestep=(timesteps * 1000).to(self.device))
        else:
            global_context.update(current_timestep=timesteps.to(self.device))

        return img_latents, noise, noisy_latents, timesteps, model_kwargs

    def _forward_pass_with_distillation(self, noisy_latents, timesteps, model_kwargs, dynamic_distill_weights=None):
        distill_loss_final = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        distill_loss_mid = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        if dynamic_distill_weights:
            w_final = dynamic_distill_weights.get("distill_final_weight", 0.0)
            w_mid = dynamic_distill_weights.get("distill_middle_weight", 0.0)
        else:
            w_final = self.config.distillation_final_weight
            w_mid = self.config.distillation_middle_weight

        if self.teacher_model is not None:
            enable_final = w_final > 0
            enable_mid = w_mid > 0

            if enable_mid:
                model_kwargs["output_hidden_states"] = True
                model_kwargs["output_hidden_states_step"] = 1
            
            with torch.autocast(device_type=self.device.type, dtype=DiffusionModelTrainer._get_dtype(self.config.mixed_precision)), torch.no_grad():
                teacher_output = self.teacher_model(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    **model_kwargs,
                )
            
            student_output = self.model(
                hidden_states=noisy_latents,
                timestep=timesteps,
                **model_kwargs,
            )
            
            if enable_mid:
                Y_s, Y_s_per = teacher_output
                Y_d, Y_d_per = student_output
                noise_pred = Y_d
            else:
                Y_s = teacher_output[0]
                Y_d = student_output[0]
                noise_pred = Y_d
            
            if enable_final:
                distill_loss_final = F.mse_loss(Y_d.float(), Y_s.float(), reduction="mean")
                distill_loss_final = distill_loss_final * w_final
            
            if enable_mid:
                assert len(Y_s_per) == len(Y_d_per), "Teacher / student layer count mismatch!"
                # for idx, (s_i, d_i) in enumerate(zip(Y_s_per, Y_d_per)):
                #     # s_i/d_i shape: (B, C, H, W)
                #     layer_loss = F.mse_loss(d_i.float(), s_i.float(), reduction="mean")
                #     distill_loss_mid += layer_loss

                # distill_loss_mid = distill_loss_mid / len(Y_s_per)
                # distill_loss_mid = distill_loss_mid * w_mid

                if len(Y_d_per) > 0:
                    pred_stack = torch.stack(Y_d_per).float()
                    target_stack = torch.stack(Y_s_per).float().detach()

                    norm_shape = (pred_stack.shape[-2], pred_stack.shape[-1])
                    
                    pred_norm = F.layer_norm(pred_stack, norm_shape)
                    target_norm = F.layer_norm(target_stack, norm_shape)

                    distill_loss_mid = F.mse_loss(pred_norm, target_norm, reduction="mean")
                    distill_loss_mid = distill_loss_mid * w_mid
        else:
            noise_pred = self.model(
                hidden_states=noisy_latents,
                timestep=timesteps,
                **model_kwargs,
            )[0]

        return noise_pred, distill_loss_final, distill_loss_mid

    def _calculate_main_loss(self, noise_pred, noise, img_latents):
        if self.model_type == "pixart-sigma":
            noise_pred = noise_pred.chunk(2, dim=1)[0]
        elif self.model_type == 'qwen-image-edit':
            noise_pred = noise_pred[:, : global_context.hidden_seq_len]

        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            # in flow-matching, we need to compute the noise target as vector field
            noise_target = noise - img_latents
        else:
            noise_target = noise

        main_loss = F.mse_loss(
            noise_pred.float(), noise_target.float(), reduction="mean"
        )

        return main_loss

    def _get_shiva_loss_config(self, global_step: int) -> dict:
        current_weights = {}

        for stage in self.processed_stage_schedule:
            current_weights = stage["weights"]
            if global_step < stage["threshold"]:
                break

        config = {}
        config["ratio_loss_weight"] = current_weights.get("ratio", [0.0, 0.0])
        config["gate_loss_weight"] = current_weights.get("gate", 0.0)
        config["aux_loss_weight"] = current_weights.get("aux", 0.0)
        distill_weights = current_weights.get("distill", [0.0, 0.0])
        config["distill_final_weight"] = distill_weights[0]
        config["distill_middle_weight"] = distill_weights[1] if len(distill_weights) > 1 else 0.0
        config["target_ratio"] = self.config.extra_config['target_ratio']

        config["gradient_accumulation_steps"] = self.config.gradient_accumulation_steps

        return config

    def _calculate_hook_loss(self, global_step: int, timestep: torch.Tensor, is_validation: bool = False):
        total_hook_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        aggregated_metrics = {}
        self.step_per_layer_metrics = {}

        current_state_buffer = self.state_buffer if not is_validation else {}

        if self.hook_type == "diffcr":
            loss_config = self.config.extra_config
            total_hook_loss = DiffCRDitBlockHook.calculate_ratio_loss(loss_config, self.device)
            aggregated_metrics["loss/diffcr_loss"] = total_hook_loss

            ratios = []
            for hook in HookManager.get_all_hooks().values():
                if isinstance(hook, DiffCRDitBlockHook):
                    ratios.append(hook.diff_ratio)
            
            if ratios:
                stacked_ratios = torch.stack(ratios)
                aggregated_metrics["diffcr/avg_ratio"] = torch.mean(stacked_ratios)
                aggregated_metrics["diffcr/min_ratio"] = torch.min(stacked_ratios)
                aggregated_metrics["diffcr/max_ratio"] = torch.max(stacked_ratios)
                
                target_r = loss_config.get("target_ratio", 0.0)
                aggregated_metrics["diffcr/target_ratio"] = torch.tensor(target_r, device=self.device)

            return total_hook_loss, aggregated_metrics

        if self.hook_type != "shiva":
            return total_hook_loss, aggregated_metrics

        loss_config = self._get_shiva_loss_config(global_step)

        # 1. Gate Loss
        gate_loss, g_metrics, per_layer_fuse_probs = shiva_metrics.calculate_gate_loss(
            loss_config, self.device
        )
        
        # 2. Image Ratio Loss
        img_ratio_loss, img_r_metrics, per_layer_ratios = (
            shiva_metrics.calculate_image_ratio_loss(loss_config, self.device, current_state_buffer)
        )
        
        # 3. Context Ratio Loss
        ctx_ratio_loss, ctx_r_metrics, per_layer_context_ratios = shiva_metrics.calculate_context_ratio_loss(loss_config, self.device)

        # 4. Context Auxiliary Loss
        ctx_aux_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        aux_weight = loss_config.get("aux_loss_weight", 0.0)
        if aux_weight > 0:
            all_aux_losses = []
            for hook in HookManager.get_all_hooks().values():
                if hasattr(hook, "context_aux_loss_accumulator") and hook.context_aux_loss_accumulator:
                    layer_aux_losses = torch.stack(hook.context_aux_loss_accumulator)
                    all_aux_losses.append(layer_aux_losses.sum())
            if all_aux_losses:
                ctx_aux_loss_raw = torch.stack(all_aux_losses).sum()
                ctx_aux_loss = ctx_aux_loss_raw * aux_weight
                aggregated_metrics["loss/context/context_aux_loss_raw"] = ctx_aux_loss_raw.detach()
                aggregated_metrics["loss/context/context_aux_loss"] = ctx_aux_loss.detach()

        context_stats = shiva_metrics.collect_context_stats(self.device)
        aggregated_metrics.update(context_stats)

        total_hook_loss = gate_loss + img_ratio_loss + ctx_ratio_loss + ctx_aux_loss

        aggregated_metrics.update(g_metrics)
        aggregated_metrics.update(img_r_metrics)
        aggregated_metrics.update(ctx_r_metrics)

        self.step_per_layer_metrics.update(
            {
                "gater/per_layer_fuse_prob": per_layer_fuse_probs,
                "ratio_policy/image_ratio/per_layer_ratios": per_layer_ratios,
                "ratio_policy/context_ratio/per_layer_context_ratios": per_layer_context_ratios,
            }
        )

        router_stats = shiva_metrics.collect_router_stats(self.device)
        aggregated_metrics.update(router_stats)

        autograd_stats = shiva_metrics.collect_autograd_stats(self.device)
        aggregated_metrics.update(autograd_stats)

        for hook in HookManager.get_all_hooks().values():
            if hasattr(hook, "clear_accumulators"):
                hook.clear_accumulators()

        return total_hook_loss, aggregated_metrics

    def _compute_loss(self, batch, global_step: int, is_validation: bool = False):
        img_latents, noise, noisy_latents, timesteps, model_kwargs = (
            self.prepare_model_inputs(batch)
        )

        noisy_latents = noisy_latents.to(self.dtype).detach()

        loss_config = self._get_shiva_loss_config(global_step)

        noise_pred, distill_loss_final, distill_loss_mid = self._forward_pass_with_distillation(
            noisy_latents, timesteps, model_kwargs, dynamic_distill_weights=loss_config
        )

        main_loss = self._calculate_main_loss(noise_pred, noise, img_latents)
        hook_loss, hook_metrics = self._calculate_hook_loss(global_step, timesteps, is_validation=is_validation)

        total_loss = main_loss + hook_loss + distill_loss_final + distill_loss_mid

        return total_loss, main_loss, hook_loss, distill_loss_final, distill_loss_mid, hook_metrics, timesteps

    def _monitor_gradient_analysis(
        self, 
        global_step: int, 
        timesteps: torch.Tensor, 
        hook_metrics: dict,
        main_loss: torch.Tensor, 
        distill_loss_final: torch.Tensor, 
        distill_loss_mid: torch.Tensor, 
        hook_loss: torch.Tensor
    ):
        debug_interval = 20
        if global_step % debug_interval != 0 or not self.accelerator.is_main_process:
            return

        print(f"\n{'='*35} [Gradient Analysis Monitor Step {global_step}] {'='*35}")
        
        # --- 1. Batch Statistics ---
        t_min, t_max = timesteps.min().item(), timesteps.max().item()
        t_std = timesteps.float().std().item()
        curr_avg_ratio = hook_metrics.get("image_ratio/avg_ratio", float('nan'))
        
        print(f"  > Batch Info:")
        print(f"    Timesteps: Min={t_min:.1f} | Max={t_max:.1f} | Std={t_std:.1f} | Sample={timesteps[:4].int().tolist()}...")
        if not isinstance(curr_avg_ratio, float) or not (curr_avg_ratio != curr_avg_ratio): # Check NaN
             print(f"    Avg Ratio: {curr_avg_ratio:.4f}")

        # --- 2. Parameter Grouping (Adapted for Additive Policy) ---
        param_groups = {
            "Router": [], 
            "Gater": [], 
            "Ratio(Time)": [],   # Time Branch (time_mlp)
            "Ratio(Layer)": [],  # Layer Branch (layer_emb + layer_mlp)
            "Ratio(Bias)": [],   # Global Bias (Anchor)
            "Fuser": [], 
            "Ctx(Pruner)": [], 
            "Ctx(Policy)": [], 
            "Base Model": []
        }
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad: continue
            
            if "router." in name: 
                param_groups["Router"].append(param)
            elif "gater_policy." in name: 
                param_groups["Gater"].append(param)
            elif "image_ratio_policy." in name:
                # Additive Policy Specifics
                if "layer_emb" in name or "layer_mlp" in name: 
                    param_groups["Ratio(Layer)"].append(param)
                elif "time_mlp" in name:
                    param_groups["Ratio(Time)"].append(param)
                elif "global_bias" in name:
                    param_groups["Ratio(Bias)"].append(param)
                else:
                    # Fallback for older names or unknown params
                    if "time" in name: param_groups["Ratio(Time)"].append(param)
                    else: param_groups["Ratio(Layer)"].append(param)
            elif "fuse_expert" in name or "fuser" in name: 
                param_groups["Fuser"].append(param)
            elif "context_pruner" in name: 
                param_groups["Ctx(Pruner)"].append(param)
            elif "context_ratio_policy" in name: 
                param_groups["Ctx(Policy)"].append(param)
            elif "_hook" not in name: 
                param_groups["Base Model"].append(param)

        # --- 3. Measure Gradients (Dry Run) ---
        # Use no_sync to avoid DDP communication overhead during this debug pass
        context_manager = self.accelerator.no_sync(self.model) if hasattr(self.accelerator, "no_sync") else torch.autograd.set_detect_anomaly(False)
        
        with context_manager:
            def measure_grads(loss_tensor, desc):
                # Handle zero/unused loss
                if loss_tensor is None or (isinstance(loss_tensor, torch.Tensor) and loss_tensor.item() == 0.0):
                    return {k: (0.0, 0.0) for k in param_groups}
                
                # Crucial: Zero grad before specific backward to isolate contribution
                self.optimizer.zero_grad(set_to_none=True)
                
                # Retain graph needed because we will call backward multiple times (for Main, Hook, Distill...)
                # and finally the real backward outside this function.
                self.accelerator.backward(loss_tensor, retain_graph=True)
                
                stats = {}
                for group_name, params in param_groups.items():
                    if not params:
                        stats[group_name] = (0.0, 0.0)
                        continue
                    
                    total_norm_sq = 0.0
                    max_val = 0.0
                    has_grad = False
                    for p in params:
                        if p.grad is not None:
                            has_grad = True
                            g_norm = p.grad.detach().data.norm(2).item()
                            total_norm_sq += g_norm ** 2
                            max_val = max(max_val, p.grad.detach().abs().max().item())
                    
                    if has_grad:
                        stats[group_name] = (total_norm_sq ** 0.5, max_val)
                    else:
                        stats[group_name] = (0.0, 0.0)
                return stats

            # Measure contributions
            main_stats = measure_grads(main_loss, "Main")
            distill_final_stats = measure_grads(distill_loss_final, "DistillFinal")
            distill_mid_stats = measure_grads(distill_loss_mid, "DistillMid")
            hook_stats = measure_grads(hook_loss, "Hook (Ratio)")
        
        # --- 4. Print Analysis Table ---
        print(f"\n  > Component Gradient Analysis (Norm / Max):")
        # Header formatting
        h_name, h_main, h_hook, h_df, h_dm, h_ratio = "Module Name", "Main Grad", "Ratio(H)", "Distill(F)", "Distill(M)", "Aux/Main"
        print(f"    {h_name:<15} | {h_main:<13} | {h_hook:<13} | {h_df:<13} | {h_dm:<13} | {h_ratio:<10}")
        print(f"    {'-'*90}")

        display_order = ["Router", "Ratio(Time)", "Ratio(Layer)", "Ratio(Bias)", "Gater", "Fuser", "Ctx(Pruner)"]
        
        for name in display_order:
            if name not in param_groups or not param_groups[name]: continue
            
            m_norm, m_max = main_stats[name]
            h_norm, h_max = hook_stats[name]
            df_norm, df_max = distill_final_stats[name]
            dm_norm, dm_max = distill_mid_stats[name]

            total_aux_norm = h_norm
            total_main_norm = m_norm + df_norm + dm_norm + 1e-9
            
            ratio_val = total_aux_norm / total_main_norm
            if m_norm < 1e-9 and total_aux_norm > 1e-9:
                ratio_str = "INF"
            else:
                ratio_str = f"{ratio_val:.2f}x"
            
            # Formatter
            def fmt(val, mx): return f"{val:.4f}/{mx:.4f}" if val > 1e-6 else "0.0"

            print(f"    {name:<15} | {fmt(m_norm, m_max):<13} | {fmt(h_norm, h_max):<13} | {fmt(df_norm, df_max):<13} | {fmt(dm_norm, dm_max):<13} | {ratio_str:<10}")

        # --- 5. Health Checks & Alerts (Based on Additive Policy Logic) ---
        rt_norm, _ = hook_stats.get("Ratio(Time)", (0,0)) # Hook loss usually drives Policy
        rl_norm, _ = hook_stats.get("Ratio(Layer)", (0,0))
        rb_norm, _ = hook_stats.get("Ratio(Bias)", (0,0))
        
        if rl_norm == 0:
             rt_norm, _ = main_stats.get("Ratio(Time)", (0,0))
             rl_norm, _ = main_stats.get("Ratio(Layer)", (0,0))

        print(f"\n  > Policy Diagnostics:")
        if rl_norm > 0:
            tl_ratio = rt_norm / rl_norm
            print(f"    T/L Energy Ratio: {tl_ratio:.4f} (Time / Layer)")
            
            if tl_ratio < 0.1:
                print(f"    ✅ [HEALTHY] Layer branch dominating ({rl_norm:.4f} vs {rt_norm:.4f}). Additive structure working.")
            elif tl_ratio > 1.0:
                print(f"    ⚠️  [ALERT] Time branch gradient is larger than Layer! ({rt_norm:.4f} > {rl_norm:.4f}). Check initialization or lr.")
            else:
                print(f"    ℹ️  [INFO] Time branch active but subordinate. (Healthy if converging)")
        
        if rb_norm > 0.1:
             print(f"    ℹ️  [INFO] Large Bias adjustment ({rb_norm:.4f}). Global target ratio is shifting.")

        # --- 6. Clean up ---
        self.optimizer.zero_grad(set_to_none=True)
        print(f"{'='*96}\n")

    def _train_step(self, batch):
        total_loss, main_loss, hook_loss, distill_loss_final, distill_loss_mid, hook_metrics, timesteps = (
            self._compute_loss(batch, self.global_step, is_validation=False)
        )

        self._monitor_gradient_analysis(
            global_step=self.global_step,
            timesteps=timesteps,
            hook_metrics=hook_metrics,
            main_loss=main_loss,
            distill_loss_final=distill_loss_final,
            distill_loss_mid=distill_loss_mid,
            hook_loss=hook_loss
        )

        self.accelerator.backward(total_loss)
        if self.config.enable_gradient_checkpointing:
            for hook in HookManager.get_all_hooks().values():
                if hasattr(hook, "clear_accumulators"):
                    hook.clear_accumulators()

        self.step_metrics = hook_metrics

        return (
            total_loss.detach(),
            main_loss.detach(),
            hook_loss.detach(),
            distill_loss_final.detach(),
            distill_loss_mid.detach(),
            timesteps.detach(),
        )

    def _set_finetuning_trainable_state(self, requires_grad: bool):        
        if self.accelerator.is_main_process:
            print(f"\n[Step {self.global_step}]: Setting finetuning parameters' requires_grad = {requires_grad}")

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        for name, param in unwrapped_model.named_parameters():
            if self.config.training_mode == "lora":                
                if "lora_" in name:
                    if param.requires_grad != requires_grad:
                        param.requires_grad = requires_grad
            elif self.config.training_mode == "finetune":
                if "_hook" not in name:
                    if param.requires_grad != requires_grad:
                        param.requires_grad = requires_grad

    def _update_learning_rate(self):
        current_lrs = None
        new_stage_idx = -1

        for idx, stage in enumerate(self.processed_stage_schedule):
            if self.global_step < stage["threshold"]:
                current_lrs = stage["lr"]
                new_stage_idx = idx
                break

        if new_stage_idx != self.current_stage_idx and new_stage_idx != -1:
            if self.accelerator.is_main_process:
                print(f"\n[Step {self.global_step}] Stage changed ({self.current_stage_idx} -> {new_stage_idx}). Resetting Optimizer State (Momentum)!")

            self.optimizer.state.clear()
            self.current_stage_idx = new_stage_idx
            global_context.update(current_stage_idx=new_stage_idx)

        if current_lrs is None:
            return

        # Strict 6-item mapping as requested
        # [Base, LoRA, Gate, Fuser, ImageRatio, ContextRatio, ContextPruner]
        
        lr_base = current_lrs[0]
        lr_lora = current_lrs[1]
        lr_gate = current_lrs[2]
        lr_fuser = current_lrs[3]
        lr_img_ratio = current_lrs[4]
        lr_ctx_ratio = current_lrs[5]
        lr_ctx_pruner = current_lrs[6]

        for param_group in self.optimizer.param_groups:
            group_name = param_group.get("name", "")

            if group_name == "base":
                param_group["lr"] = lr_base
            elif group_name == "lora":
                param_group["lr"] = lr_lora
            elif group_name == "gate_policy":
                param_group["lr"] = lr_gate
            elif group_name == "image_ratio_policy":
                param_group["lr"] = lr_img_ratio
            elif group_name == "fuser":
                param_group["lr"] = lr_fuser
            elif group_name == "context_ratio_policy":
                param_group["lr"] = lr_ctx_ratio
            elif group_name == "context_pruner":
                param_group["lr"] = lr_ctx_pruner

    def _update_temperature(self):
        current_Ts = None
        current_Tk = None

        for stage in self.processed_stage_schedule:
            if self.global_step < stage["threshold"]:
                temp_config = stage["temp"]
                progress = 0.0
                if stage["threshold"] != float('inf'):
                    start_step = stage["start_step"]
                    end_step = stage["threshold"]
                    if end_step > start_step:
                        progress = (self.global_step - start_step) / (end_step - start_step)
                        progress = max(0.0, min(1.0, progress))

                if len(temp_config) == 4:
                    # [start_Ts, end_Ts, start_Tk, end_Tk]
                    s_ts, e_ts, s_tk, e_tk = temp_config
                    current_Ts = s_ts + (e_ts - s_ts) * progress
                    current_Tk = s_tk + (e_tk - s_tk) * progress
                elif len(temp_config) == 2:
                    # [start, end] -> Ts=Tk
                    s_t, e_t = temp_config
                    val = s_t + (e_t - s_t) * progress
                    current_Ts = val
                    current_Tk = val
                else:
                    current_Ts = 5.0
                    current_Tk = 5.0

                break

        if current_Ts is None or current_Tk is None:
            return

        ts_val = torch.tensor(current_Ts, device=self.device, dtype=self.dtype)
        tk_val = torch.tensor(current_Tk, device=self.device, dtype=self.dtype)

        for hook in HookManager.get_all_hooks().values():
            # Image Pruner Expert
            if hasattr(hook, "prune_expert") and hook.prune_expert is not None:
                if hasattr(hook.prune_expert, "Ts"):
                    hook.prune_expert.Ts = ts_val
                if hasattr(hook.prune_expert, "Tk"):
                    hook.prune_expert.Tk = tk_val
            
            # Context Pruner
            if hasattr(hook, "context_pruner") and hook.context_pruner is not None:
                if hasattr(hook.context_pruner, "Ts"):
                    hook.context_pruner.Ts = ts_val.float()
                if hasattr(hook.context_pruner, "Tk"):
                    hook.context_pruner.Tk = tk_val.float()

            pass

    def _on_update_step_end(self, gathered_losses, progress_bar, batch=None):
        log_trigger = self.log_trigger
        validate_trigger = self.validate_trigger
        inference_trigger = self.inference_trigger
        save_trigger = self.save_trigger

        val_metric = None
        if validate_trigger and self.global_step % validate_trigger == 0:
            val_metric = self.validate(self.global_step)

            if self.config.save_best and val_metric is not None:
                self.save_specific_best(
                    metric_name="val_main_loss", 
                    metric_value=val_metric, 
                    higher_is_better=False
                )

        if inference_trigger and self.global_step % inference_trigger == 0:
            inf_metrics = self.run_inference_validation(self.global_step)

            if self.config.save_best and inf_metrics:
                for name, value in inf_metrics.items():
                    self.save_specific_best(
                        metric_name=name, 
                        metric_value=value, 
                        higher_is_better=True
                    )
            self.accelerator.wait_for_everyone()

        if save_trigger and self.global_step % save_trigger == 0:
            self.maybe_save_periodically()

        metrics_to_log_tensors = self.step_metrics.copy()

        # Extract LRs for logging
        lrs_to_log = {}
        for pg in self.optimizer.param_groups:
            name = pg.get("name", "unknown")
            lrs_to_log[f"train/lr_{name}"] = torch.tensor(pg["lr"], device=self.device)
        
        metrics_to_log_tensors.update(lrs_to_log)

        current_Ts = 0.0
        current_Tk = 0.0
        for hook in HookManager.get_all_hooks().values():
            if hasattr(hook, "prune_expert") and hook.prune_expert:
                current_Ts = hook.prune_expert.Ts.item()
                current_Tk = hook.prune_expert.Tk.item()
                break
        
        context_Ts = 0.0
        context_Tk = 0.0
        for hook in HookManager.get_all_hooks().values():
            if hasattr(hook, "context_pruner") and hook.context_pruner:
                context_Ts = hook.context_pruner.Ts.item()
                context_Tk = hook.context_pruner.Tk.item()
                break

        metrics_to_log_tensors.update({
            "loss/train/total_loss": gathered_losses["total"],
            "loss/train/main_loss": gathered_losses["main"],
            "loss/train/extra_loss": gathered_losses["hook"],
            "loss/train/distill_loss_final": gathered_losses["distill_final"],
            "loss/train/distill_loss_mid": gathered_losses["distill_mid"],
            "train/Ts": torch.tensor(current_Ts, device=self.device),
            "train/Tk": torch.tensor(current_Tk, device=self.device),
            "train/context_Ts": torch.tensor(context_Ts, device=self.device),
            "train/context_Tk": torch.tensor(context_Tk, device=self.device),
        })

        if self.config.training_mode == 'lora':
            lora_params = [
                p for n, p in self.model.named_parameters() if "lora_" in n
            ]
            if lora_params:
                lora_norm = torch.norm(
                    torch.stack([torch.norm(p.detach()) for p in lora_params])
                )
                lora_grads = [
                    torch.norm(p.grad.detach())
                    for p in lora_params
                    if p.grad is not None
                ]
                lora_grad_norm = torch.norm(torch.stack(lora_grads)) if lora_grads else torch.tensor(0.0, device=self.device)
                metrics_to_log_tensors.update({
                    "lora/param_norm": lora_norm, 
                    "lora/grad_norm": lora_grad_norm, 
                })
        elif self.config.training_mode == "finetune":
            all_trainable_params = [
                p for p in self.model.parameters() if p.requires_grad
            ]
            if all_trainable_params:
                param_norm = torch.norm(
                    torch.stack([torch.norm(p.detach()) for p in all_trainable_params])
                )
                param_grads = [
                    torch.norm(p.grad.detach())
                    for p in all_trainable_params
                    if p.grad is not None
                ]
                grad_norm = torch.norm(torch.stack(param_grads)) if param_grads else torch.tensor(0.0, device=self.device)
                metrics_to_log_tensors.update({
                    "finetune/param_norm": param_norm, 
                    "finetune/grad_norm": grad_norm, 
                })

        if self.hook_type == "shiva":
            shiva_grad_metrics = {}
            # Helper to collect grads
            grads_router = []
            grads_gate = []
            grads_img_ratio = []
            grads_ctx_ratio = []
            grads_ctx_pruner = []

            unwrapped_model = self.accelerator.unwrap_model(self.model)
            for name, param in unwrapped_model.named_parameters():
                if param.grad is None:
                    continue
                gn = torch.norm(param.grad.detach())
                
                if "router." in name: grads_router.append(gn)
                elif "gater_policy." in name: grads_gate.append(gn)
                elif "image_ratio_policy." in name: grads_img_ratio.append(gn)
                elif "context_ratio_policy." in name: grads_ctx_ratio.append(gn)
                elif "context_pruner." in name: grads_ctx_pruner.append(gn)

            if grads_router: shiva_grad_metrics["grad/router_avg_norm"] = torch.stack(grads_router).mean()
            if grads_gate: shiva_grad_metrics["grad/gater_policy_avg_norm"] = torch.stack(grads_gate).mean()
            if grads_img_ratio: shiva_grad_metrics["grad/image_ratio_policy_avg_norm"] = torch.stack(grads_img_ratio).mean()
            if grads_ctx_ratio: shiva_grad_metrics["grad/context_ratio_policy_avg_norm"] = torch.stack(grads_ctx_ratio).mean()
            if grads_ctx_pruner: shiva_grad_metrics["grad/context_pruner_avg_norm"] = torch.stack(grads_ctx_pruner).mean()

            metrics_to_log_tensors.update(shiva_grad_metrics)

        metrics_local_only = {}
        metrics_to_sync = {}

        for key, value in metrics_to_log_tensors.items():
            if key.startswith("topk/"):
                metrics_local_only[key] = value

            else:
                metrics_to_sync[key] = value

        if metrics_local_only:
            self.metric_manager.update(metrics_local_only, gather=False)

        if metrics_to_sync:
            self.metric_manager.update(metrics_to_sync, gather=False)

        if self.accelerator.is_main_process:
            stateful_metrics_to_log = {}
            if self.config.training_mode == 'lora':
                lora_params = [
                    p for n, p in self.model.named_parameters() if "lora_" in n
                ]
                if self.prev_lora_state is not None and lora_params:
                    deltas = [
                        torch.norm(p.detach().cpu() - self.prev_lora_state[i])
                        for i, p in enumerate(lora_params)
                    ]
                    lora_delta = torch.mean(torch.stack(deltas))
                else:
                    lora_delta = 0.0
                self.prev_lora_state = [p.detach().cpu().clone() for p in lora_params]
                stateful_metrics_to_log.update({
                    "lora/param_delta": lora_delta
                })
            elif self.config.training_mode == "finetune":
                all_trainable_params = [
                    p for p in self.model.parameters() if p.requires_grad
                ]
                if self.prev_finetune_state is not None and all_trainable_params:
                    deltas = [
                        torch.norm(p.detach().cpu() - self.prev_finetune_state[i])
                        for i, p in enumerate(all_trainable_params)
                    ]
                    param_delta = torch.mean(torch.stack(deltas))
                else:
                    param_delta = 0.0

                self.prev_finetune_state = [
                    p.detach().cpu().clone() for p in all_trainable_params
                ]
                stateful_metrics_to_log.update({
                    "finetune/param_delta": param_delta
                })

            if stateful_metrics_to_log:
                self.metric_manager.update(stateful_metrics_to_log)

            if self.global_step % log_trigger == 0:
                postfix_dict = {
                    "loss": f"{self.metric_manager.get_metric('loss/train/total_loss'):.4f}",
                    "main_loss": f"{self.metric_manager.get_metric('loss/train/main_loss'):.4f}",
                    "extra_loss": f"{self.metric_manager.get_metric('loss/train/extra_loss'):.4f}",
                    "step": self.global_step,
                }

                if self.hook_type == "shiva":
                    # Use get_metric with default 0.0 to avoid errors if mode is off
                    img_rl = self.metric_manager.get_metric('loss/image_ratio/ratio_loss', 0.0)
                    ctx_rl = self.metric_manager.get_metric('loss/context_ratio/ratio_loss', 0.0)
                    postfix_dict["r_loss"] = f"I:{img_rl:.3f}/C:{ctx_rl:.3f}"

                progress_bar.set_postfix(postfix_dict)
                self._log_router_visualization(batch=batch)
                self._log_context_visualization(batch=batch)
                self._log_loss_distribution(self.main_viz_buffer, "main_loss")
                self._log_loss_distribution(self.distill_final_viz_buffer, "distill_final_loss")
                self._log_loss_distribution(self.distill_mid_viz_buffer, "distill_mid_loss")

            self.metric_manager.log_step(self.global_step)
            if hasattr(self, "step_per_layer_metrics"):
                for key, value_dict in self.step_per_layer_metrics.items():
                    if value_dict:
                        self.logger.add_scalars(key, value_dict, self.global_step)

    def _log_policy_visualization(self):
        if not self.accelerator.is_main_process:
            return

        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            import numpy as np
        except ImportError:
            return

        from shiva_dit.hook import HookManager
        hooks = list(HookManager.get_all_hooks().values())
        if not hooks:
            return
        hook = hooks[0]
        total_layers = getattr(hook, "total_layers", 28)
        
        valid_layers = [i for i in range(total_layers) if i not in self.skip_layer_indices]
        if len(valid_layers) == 0: return
        
        steps = 50 # Resolution for Time axis
        
        # Crucial: Match the input domain of the model (0-1 vs 0-1000)
        if self.config.model_type in ["flux-dev", "qwen-image", "qwen-image-edit", "flux-kontext"]:
            # Flow Matching: 0.0 -> 1.0
            t_model_input = torch.linspace(0, 1, steps, device=self.device, dtype=self.dtype)
            t_display_axis = np.linspace(0, 1, steps)
            t_label = "Time (0.0=Start -> 1.0=End)"
        else:
            # Standard DiT / SD3: 0 -> 1000 (usually)
            # Some SD3 implementations involve 0-1, but Diffusers standard is often 0-1000 for embeddings
            t_model_input = torch.linspace(0, 1000, steps, device=self.device, dtype=self.dtype)
            t_display_axis = np.linspace(0, 1000, steps)
            t_label = "Timestep (0 -> 1000)"

        # Prepare Grid Tensors
        valid_layers_tensor = torch.tensor(valid_layers, device=self.device, dtype=self.dtype)
        
        # Meshgrid: (Steps, Layers)
        t_grid, l_grid = torch.meshgrid(t_model_input, valid_layers_tensor, indexing='ij')
        t_flat = t_grid.reshape(-1)
        l_flat = l_grid.reshape(-1)

        temb_input = None
        try:
            with torch.no_grad():
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                
                # Logic to find the time embedder
                time_embedder = None
                
                # Case A: Flux / SD3 (time_text_embed)
                if hasattr(unwrapped_model, "time_text_embed"):
                    # Flux usually projects scalar t -> vector
                    # Note: We manually invoke the submodule to avoid full forward pass
                    emb_module = unwrapped_model.time_text_embed
                    if hasattr(emb_module, "time_proj") and hasattr(emb_module, "timestep_embedder"):
                        t_proj = emb_module.time_proj(t_flat)
                        temb_input = emb_module.timestep_embedder(t_proj.to(dtype=self.dtype))
                    elif hasattr(emb_module, "forward"):
                        # Try calling it directly if it accepts just timestep
                        # Flux time_text_embed usually takes (timestep, pooled_text, text_seq_len)
                        # This might be tricky without text. 
                        # Fallback: Assume we have a helper or specific projection available
                        pass

        except Exception as e:
            print(f"[Viz] ⚠️ Warning: Failed to extract Time Embedding: {e}")
            # If Policy is Additive and Zero-Inited, maybe we don't strictly need temb for visualization 
            # if we just want to see the Layer component. But normally we do.

        policies_to_viz = [
            ("image_ratio", "image_ratio_policy", "Keep Ratio (Sigmoid)"),
            # ("context_ratio", "context_ratio_policy", "Keep Ratio (Sigmoid)"),
            # ("gater", "gater_policy", "Fuse Probability"),
        ]

        print(f"\n{'='*30} [Policy Visualization Report] {'='*30}")
        
        for alias, attr_name, metric_name in policies_to_viz:
            policy = getattr(hook, attr_name, None)
            if policy is None: continue

            # Determine Input
            p_param = next(policy.parameters())
            policy_input = None
            
            need_vector = False
            if hasattr(policy, "time_mlp") and isinstance(policy.time_mlp[0], torch.nn.Linear):
                if policy.time_mlp[0].in_features > 1:
                    need_vector = True
            
            if need_vector:
                if temb_input is None:
                    print(f"  [Skip] {alias}: Time embedding vector missing.")
                    continue
                policy_input = temb_input.to(dtype=p_param.dtype)
            else:
                # Fallback to scalar input if architecture allows (e.g. simple lookup)
                policy_input = t_flat.unsqueeze(-1).to(dtype=p_param.dtype)

            # --- Forward ---
            with torch.no_grad():
                logits = policy(policy_input, l_flat.long())
                
                if "gater" in alias:
                    probs = torch.softmax(logits, dim=-1)
                    values = probs[:, 1] # Fuse prob
                else:
                    values = torch.sigmoid(logits).reshape(-1)

            # --- Data Processing ---
            # Shape: (Steps, Valid_Layers)
            grid_values = values.reshape(steps, len(valid_layers)).float().cpu().numpy()
            
            # Stats
            g_mean = grid_values.mean()
            g_min, g_max = grid_values.min(), grid_values.max()
            
            # Marginal Stats
            layer_profile = grid_values.mean(axis=0) # Avg over time
            time_profile = grid_values.mean(axis=1)  # Avg over layers
            
            layer_var = layer_profile.std()
            time_var = time_profile.std()

            # --- Detailed Console Log ---
            print(f"\n  > Target: {alias.upper()}")
            print(f"    Global Stats : Mean={g_mean:.4f} | Range=[{g_min:.4f}, {g_max:.4f}]")
            print(f"    Variance     : Spatial(Layer)={layer_var:.4f} vs Temporal(Time)={time_var:.4f}")
            
            if time_var < 0.01:
                print(f"    Observation  : 📉 Temporal dimension is FLAT (Stable).")
            else:
                print(f"    Observation  : 📈 Temporal dimension shows variation.")

            # Print ASCII Sparklines (Simplified)
            print(f"    Layer Profile (Avg over T):")
            print(f"      {np.array2string(layer_profile, precision=3, suppress_small=True)}")
            print(f"    Time Profile (Avg over L, sampled):")
            # Downsample time profile for printing
            t_sample_indices = np.linspace(0, steps-1, 10).astype(int)
            print(f"      {np.array2string(time_profile[t_sample_indices], precision=3)}")

            # --- Plotting ---
            fig = plt.figure(figsize=(12, 9))
            gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3], wspace=0.05, hspace=0.05)

            # 1. Main Heatmap (Bottom-Left)
            ax_main = fig.add_subplot(gs[1, 0])
            im = ax_main.imshow(grid_values, aspect='auto', origin='lower',
                                extent=[0, len(valid_layers)-1, 0, 1],
                                vmin=0, vmax=1, cmap='viridis')
            ax_main.set_xlabel('Transformer Layer Index (Mapped)', fontsize=10)
            ax_main.set_ylabel(t_label, fontsize=10)
            ax_main.set_title(f"Policy Landscape", fontsize=10)

            # 2. Top: Layer Profile (Top-Left)
            ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
            ax_top.plot(np.arange(len(valid_layers)), layer_profile, marker='o', markersize=3, linestyle='-', color='blue')
            ax_top.set_ylim(0, 1)
            ax_top.grid(True, alpha=0.3)
            ax_top.set_ylabel("Avg Ratio")
            ax_top.set_title("Spatial Distribution (Layer)", fontsize=10)
            plt.setp(ax_top.get_xticklabels(), visible=False)

            # 3. Right: Time Profile (Bottom-Right)
            ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
            # Plot y against x to match heatmap orientation
            ax_right.plot(time_profile, np.linspace(0, 1, steps), marker='o', markersize=3, linestyle='-', color='red')
            ax_right.set_xlim(0, 1)
            ax_right.grid(True, alpha=0.3)
            ax_right.set_xlabel("Avg Ratio")
            ax_right.set_title("Temporal Distribution", fontsize=10)
            plt.setp(ax_right.get_yticklabels(), visible=False)

            # Colorbar
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            fig.colorbar(im, cax=cbar_ax, label=metric_name)

            plt.suptitle(f'Shiva Policy Analysis: {alias.upper()}\nSpatial Var: {layer_var:.4f} | Temporal Var: {time_var:.4f}', fontsize=14)
            
            # Log
            self.logger.log_figure(f"policy/{alias}_heatmap", fig, self.global_step)
            plt.close(fig)
            print(f"    -> Plot saved to Tensorboard.")

        print(f"{'='*90}\n")

    def _log_router_visualization(self, batch=None):
        if not self.accelerator.is_main_process:
            return
        
        from shiva_dit.hook import HookManager
        all_hooks = HookManager.get_all_hooks()
        hooks = list(HookManager.get_all_hooks().values())
        if hasattr(hooks[0], 'shiva_mode') and hooks[0].shiva_mode == 'image':
            if batch is not None and "pixel_values" in batch:
                try:
                    import torchvision

                    try:
                        import cv2
                        import numpy as np
                    except ImportError:
                        cv2 = None

                    images = batch["pixel_values"].detach().cpu().float()

                    timesteps = global_context.current_timestep
                    if timesteps is not None:
                        timesteps = timesteps.detach().cpu().flatten()

                    images = (images + 1.0) / 2.0
                    images = images.clamp(0.0, 1.0)

                    annotated_images = []
                    for i in range(images.shape[0]):
                        img_tensor = images[i]  # (C, H, W)

                        if cv2 is not None and timesteps is not None and i < len(timesteps):
                            # Tensor (C, H, W) -> Numpy (H, W, C)
                            img_np = img_tensor.permute(1, 2, 0).numpy()
                            img_np = (img_np * 255).astype(np.uint8)
                            img_np = np.ascontiguousarray(img_np)

                            t_val = timesteps[i].item()
                            if t_val <= 1.0:
                                text = f"t={t_val:.3f}"
                            else:
                                text = f"t={int(t_val)}"

                            cv2.putText(
                                img_np,
                                text,
                                (5, 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 0, 0),
                                2,
                            )
                            cv2.putText(
                                img_np,
                                text,
                                (5, 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 255),
                                1,
                            )

                            # Numpy (H, W, C) -> Tensor (C, H, W)
                            img_tensor = (
                                torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
                            )

                        annotated_images.append(img_tensor)

                    if annotated_images:
                        images = torch.stack(annotated_images)

                    grid_input = torchvision.utils.make_grid(
                        images,
                        nrow=int(len(images) ** 0.5) or 1,
                        normalize=False,
                        pad_value=1.0,
                    )

                    self.logger.log_image(
                        "router_viz/input_image", grid_input, self.global_step
                    )
                except Exception as e:
                    print(f"Failed to log input image: {e}")

            num_layers = global_context.num_layers
            if global_context.num_single_layers is not None:
                num_layers += global_context.num_single_layers
            target_layers = range(num_layers)

            H = global_context.hidden_seq_height
            W = global_context.hidden_seq_width

            for layer_idx in target_layers:
                target_hook = None
                for name, hook in all_hooks.items():
                    if hook.layer_idx == layer_idx and hasattr(hook, "last_token_scores") and getattr(hook, "last_sel_idx", None) is not None:
                        target_hook = hook
                        break

                if target_hook is not None and target_hook.last_token_scores is not None:
                    heatmap_tensor = shiva_metrics.generate_router_heatmap(
                        target_hook.last_token_scores,
                        target_hook.last_sel_idx,
                        H, W
                    )

                    if heatmap_tensor is not None:
                        try:
                            import torchvision
                            grid_img_tensor = torchvision.utils.make_grid(
                                heatmap_tensor, nrow=heatmap_tensor.shape[0], normalize=False, pad_value=0.5
                            )
                            self.logger.log_image(f"router_viz/layer_{layer_idx:02d}", grid_img_tensor, self.global_step)
                        except ImportError:
                            self.logger.log_image(f"router_viz/layer_{layer_idx:02d}", heatmap_tensor[0], self.global_step)

                    target_hook.last_token_scores = None
                    target_hook.last_sel_idx = None

    def _log_context_visualization(self, batch):
        if not self.accelerator.is_main_process:
            return
        if batch is None:
            return

        from shiva_dit.hook import HookManager
        all_hooks = HookManager.get_all_hooks()
        hooks = list(HookManager.get_all_hooks().values())
        if hasattr(hooks[0], 'shiva_mode') and hooks[0].shiva_mode == 'context':
            input_ids = None
            if "input_ids" in batch:
                val = batch["input_ids"]
                if isinstance(val, list) or isinstance(val, tuple):
                    input_ids = val[-1]
                else:
                    input_ids = val
            
            if input_ids is None:
                return

            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids[0].detach().cpu()
            
            tokenizer = self.tokenizers[-1]

            try:
                from shiva_dit.utils.text_viz import render_token_importance
            except ImportError:
                return

            num_layers = global_context.num_layers
            if global_context.num_single_layers is not None:
                num_layers += global_context.num_single_layers
            target_layers = range(num_layers)

            for layer_idx in target_layers:
                target_hook = None
                for name, hook in all_hooks.items():
                    if hook.layer_idx == layer_idx and hasattr(hook, "last_context_mask"):
                        target_hook = hook
                        break
                
                if target_hook is not None and target_hook.last_context_mask is not None:
                    try:
                        mask = target_hook.last_context_mask[0].cpu()
                        viz_img = render_token_importance(
                            tokenizer, 
                            input_ids, 
                            mask=mask,
                            width=1000
                        )
                        
                        self.logger.log_image(f"context_viz/layer_{layer_idx:02d}", viz_img, self.global_step)
                    
                    except Exception as e:
                        print(f"Failed to log context viz for layer {layer_idx}: {e}")
                    finally:
                        target_hook.last_context_mask = None

    def _log_loss_distribution(self, buffer, prefix):
        if not self.accelerator.is_main_process:
            return
        
        if len(buffer) < 1000: 
            return

        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            return

        data = np.array(buffer)
        x = data[:, 0] # Time
        y = data[:, 1] # Loss
        
        buffer = []

        fig, ax = plt.subplots(figsize=(10, 6))
        sc = ax.scatter(x, y, s=5, alpha=0.6, c=y, cmap='viridis')
        try:
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            y_sorted = y[sort_idx]
            window_size = 50
            if len(x_sorted) > window_size:
                y_smooth = np.convolve(y_sorted, np.ones(window_size)/window_size, mode='valid')
                ax.plot(x_sorted[window_size//2 : -window_size//2 + 1], y_smooth, 'r-', linewidth=2, label='Trend')
        except Exception:
            pass

        ax.set_xlabel('Timestep (0 -> 1000)')
        ax.set_ylabel('Loss (Middle)')
        ax.set_title(f'Loss Distribution (Last {len(x)} steps)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.colorbar(sc, ax=ax, label='Loss Magnitude')

        self.logger.log_figure(f"debug/{prefix}_vs_time", fig, self.global_step)
        plt.close(fig)

    def _log_trainable_parameters(self, description: str = ""):
        if not self.accelerator.is_main_process:
            return

        trainable_params = 0
        hook_params = 0
        lora_params = 0
        all_params = 0
        trainable_param_names = []

        for name, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                trainable_param_names.append(name)
                if "_hook" in name:
                    hook_params += param.numel()
                if "lora_" in name:
                    lora_params += param.numel()

        print(f"\n--- Trainable Parameters ({description}) ---")
        print(f"  Currently trainable parameters: {trainable_params:,}")
        print(f"  Total parameters:   {all_params:,}")
        print(f"  Hook parameters:   {hook_params:,}")
        print(f"  LoRA parameters:   {lora_params:,}")
        print(f"  Trainable ratio:   {100 * trainable_params / all_params:.4f}%")

        if len(trainable_param_names) > 0 and len(trainable_param_names) < 15:
            print(f"  Trainable parameters: {trainable_param_names}")
        elif len(trainable_param_names) > 0:
            print(f"  Trainable parameters: ['{trainable_param_names[0]}', ..., '{trainable_param_names[-1]}']")
        else:
            print(f"  Trainable parameters: None")
        print(f"--------------------------------------------------\n")

    @torch.no_grad()
    def validate(self, global_step):
        if self.val_dataloader is None or not self.config.run_validation:
            return

        if self.accelerator.is_main_process:
            print(f"\nRunning validation at step {global_step}...")

        if self.hook_type == "shiva":
            for hook in HookManager.get_all_hooks().values():
                if hasattr(hook, "clear_accumulators"): 
                    hook.is_validating = True

        self.model.eval()

        all_total_losses = []
        all_main_losses = []
        all_hook_losses = []
        all_distill_losses = []

        all_hook_metrics = []

        val_progress_bar = tqdm(
            self.val_dataloader,
            desc=f"Validation Step {global_step}",
            leave=False,
            total=len(self.val_dataloader),
            disable=not self.accelerator.is_local_main_process,
        )

        for batch in val_progress_bar:
            try:
                total_loss, main_loss, hook_loss, distill_loss_final, distill_loss_mid, hook_metrics, _ = self._compute_loss(
                    batch, global_step, is_validation=True
                )

                all_total_losses.append(total_loss)
                all_main_losses.append(main_loss)
                all_hook_losses.append(hook_loss)
                all_distill_losses.append(distill_loss_final + distill_loss_mid)
                all_hook_metrics.append(hook_metrics) 
            except Exception as e:
                if self.accelerator.is_main_process:
                    print(f"Error during validation batch: {e}")
                continue

        avg_val_loss = None
        if len(self.val_dataloader) > 0:
            avg_val_loss = (
                self.accelerator.gather(torch.stack(all_total_losses)).mean().item()
            )
            avg_main_loss = (
                self.accelerator.gather(torch.stack(all_main_losses)).mean().item()
            )
            avg_hook_loss = (
                self.accelerator.gather(torch.stack(all_hook_losses)).mean().item()
            )
            avg_distill_loss = (
                self.accelerator.gather(torch.stack(all_distill_losses)).mean().item()
            )

            if self.accelerator.is_main_process:
                print(f"--- Validation Results (Step {global_step}) ---")
                print(f"  Avg Total Loss: {avg_val_loss:.4f}")
                print(f"  Avg Main Loss:  {avg_main_loss:.4f}")
                print(f"  Avg Hook Loss:  {avg_hook_loss:.4f}")
                print(f"  Avg Distill Loss: {avg_distill_loss:.4f}")
                print(f"--------------------------------------")

                val_metrics_to_log = {}
                if all_hook_metrics:
                    keys_to_log = [
                        "loss/image_ratio/ratio_loss",
                        "loss/context_ratio/ratio_loss",
                        "loss/gater/gate_loss",
                        "ratio_policy/image_ratio/avg_ratio",
                        "ratio_policy/context_ratio/avg_ratio",
                        "gater/avg_fuse_prob"
                    ]

                    avg_hook_metrics = {}
                    for key in keys_to_log:
                        if key in all_hook_metrics[0]: 
                            try:
                                avg_metric = sum(m[key] for m in all_hook_metrics) / len(all_hook_metrics)
                                avg_hook_metrics[f"validation/{key}"] = avg_metric
                            except Exception:
                                pass
                    val_metrics_to_log.update(avg_hook_metrics)
                
                self.logger.log_scalars(
                    {
                        "validation/total_loss": avg_val_loss,
                        "validation/main_loss": avg_main_loss,
                        "validation/hook_loss": avg_hook_loss,
                        "validation/distill_loss": avg_distill_loss,
                    },
                    step=global_step
                )

                if val_metrics_to_log:
                    self.logger.log_scalars(val_metrics_to_log, step=global_step)

        elif self.accelerator.is_main_process:
            print(
                f"Validation dataloader is empty or validation failed, skipping logging."
            )

        if self.accelerator.is_main_process:
            self._log_policy_visualization()

        self.model.train()
        if self.hook_type == "shiva":
            for hook in HookManager.get_all_hooks().values():
                if hasattr(hook, "is_validating"):
                    hook.is_validating = False

        return avg_main_loss if "avg_main_loss" in locals() else None

    @torch.no_grad()
    def run_inference_validation(self, global_step):
        if not self.config.run_inference_validation or not self.config.inference_metadata_file:
            return {}

        prompts = []
        if self.accelerator.is_main_process:
            print(f"\nRunning inference validation from file: {self.config.inference_metadata_file}...")
            try:
                with open(self.config.inference_metadata_file, "r", encoding="utf-8") as f:
                    for line in f:
                        item = json.loads(line)
                        if "text" in item:
                            prompts.append(item["text"])
            except Exception as e:
                print(f"  Failed to read metadata file: {e}")

        broadcast_data = [prompts]
        broadcast_object_list(broadcast_data, from_process=0)
        prompts = broadcast_data[0]

        if not prompts:
            return {}

        original_model_in_pipe = None
        original_scheduler = None
        model_attr_name = self.model_attrs[0]

        inference_metrics = {}
        generated_images_tensors = [] 
        generated_images_pil = []     
        target_prompts = []

        if self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            original_model_in_pipe = getattr(self.pipe, model_attr_name)
            setattr(self.pipe, model_attr_name, unwrapped_model)

            inference_generator = torch.Generator(device=self.device).manual_seed(self.config.inference_seed)
            original_scheduler = copy.deepcopy(self.scheduler)

            self.model.eval()

            sample_dir = os.path.join(
                self.config.output_dir,
                self.config.inference_output_dir,
                f"step_{global_step}",
            )
            os.makedirs(sample_dir, exist_ok=True)

            for i, prompt in enumerate(prompts):
                if self.accelerator.is_main_process:
                    try:
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()

                        with self.accelerator.autocast():
                            image = self.pipe(
                                prompt=prompt,
                                generator=inference_generator,
                                num_inference_steps=self.config.inference_num_inference_steps,
                            ).images[0]

                        safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt)[:50]
                        image.save(os.path.join(sample_dir, f"sample_{i+1:03d}_{safe_prompt}.png"))
                        tag = f"inference_sample/{i+1:03d}_{safe_prompt}"
                        self.logger.log_image(tag, image, global_step)

                        generated_images_pil.append(image)
                        target_prompts.append(prompt)
                        if CLIPScore:
                            img_tensor = TF.to_tensor(image).to(self.device)
                            generated_images_tensors.append(img_tensor)

                    except Exception as e:
                        print(f"  Failed to generate image for prompt: '{prompt}'. Error: {e}")

        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            if generated_images_tensors:
                imgs_stack_float = torch.stack(generated_images_tensors) # [0, 1] float
                imgs_stack_uint8 = (imgs_stack_float * 255).to(dtype=torch.uint8) # [0, 255] uint8

                if CLIPScore and 'clip' in self.evaluators:
                    try:
                        self.evaluators['clip'].update(imgs_stack_uint8, target_prompts)
                        clip_val = self.evaluators['clip'].compute().item()
                        self.evaluators['clip'].reset()
                        inference_metrics['clip_score'] = clip_val
                        print(f"  >> CLIP Score: {clip_val:.4f}")
                    except Exception as e:
                        print(f"  Error calculating CLIP Score: {e}")

                if KernelInceptionDistance and 'kid' in self.evaluators:
                    try:
                        if not self.kid_stats_ready and self.val_dataloader is not None:
                            real_size = 128
                            print(f"  Computing KID real stats..., {real_size} imgs")
                            for i, batch in enumerate(self.val_dataloader):
                                if i >= real_size: break
                                real_imgs = ((batch["pixel_values"].to(self.device) + 1) * 127.5).clamp(0, 255).to(dtype=torch.uint8)
                                self.evaluators['kid'].update(real_imgs, real=True)
                            self.kid_stats_ready = True

                        self.evaluators['kid'].update(imgs_stack_uint8, real=False)
                        kid_mean, kid_std = self.evaluators['kid'].compute()
                        self.evaluators['kid'].reset()

                        inference_metrics['kid_mean'] = kid_mean.item()
                        print(f"  >> KID: {kid_mean.item():.5f} (±{kid_std.item():.5f})")
                    except Exception as e:
                        print(f"  Error calculating KID: {e}")

                if CLIPImageQualityAssessment is not None and 'clip_iqa' in self.evaluators:
                    try:
                        # Update expects float [0, 1]
                        self.evaluators['clip_iqa'].update(imgs_stack_float)
                        iqa_res = self.evaluators['clip_iqa'].compute()
                        self.evaluators['clip_iqa'].reset()

                        if isinstance(iqa_res, dict):
                            for k, v in iqa_res.items():
                                score = v.mean().item()
                                inference_metrics[f'clip_iqa_{k}'] = score
                                print(f"  >> CLIP-IQA ({k}): {score:.4f}")
                        else:
                            score = iqa_res.mean().item()
                            inference_metrics['clip_iqa_quality'] = score
                            print(f"  >> CLIP-IQA (quality): {score:.4f}")
                    except Exception as e:
                        print(f"  Error calculating CLIP-IQA: {e}")

            if RM and generated_images_pil and 'image_reward' in self.evaluators:
                try:
                    rewards = []
                    for img, txt in zip(generated_images_pil, target_prompts):
                        score = self.evaluators['image_reward'].score(txt, img)
                        rewards.append(score)
                    if rewards:
                        avg_reward = sum(rewards) / len(rewards)
                        inference_metrics['image_reward'] = avg_reward
                        print(f"  >> ImageReward: {avg_reward:.4f}")
                except Exception as e:
                    print(f"  Error calculating ImageReward: {e}")

            if inference_metrics:
                metrics_to_log = {f"inference/{k}": v for k, v in inference_metrics.items()}
                self.logger.log_scalars(metrics_to_log, global_step)

            setattr(self.pipe, model_attr_name, original_model_in_pipe)
            self.scheduler = original_scheduler
            print(f"  Restored original model to pipe.{model_attr_name}")
            self.model.train()

        return inference_metrics

    def _try_resume(self):
        self.global_step = 0
        self.epoch = 0
        self.best_val_metric = None

        ckpt_dir = self.config.output_dir
        latest_ckpt_dir = None
        resumed = False

        if self.accelerator.is_main_process:
            if os.path.isdir(ckpt_dir):
                ckpt_dirs = [d for d in glob.glob(os.path.join(ckpt_dir, "checkpoint_step_*")) if os.path.isdir(d)]
                if ckpt_dirs:
                    def extract_step_dir(path):
                        match_step = re.search(r"step_(\d+)$", os.path.basename(path))
                        if match_step:
                            if os.path.exists(os.path.join(path, "trainer_state.json")):
                                return int(match_step.group(1))
                        return -1

                    valid_ckpt_dirs = [d for d in ckpt_dirs if extract_step_dir(d) != -1]
                    if valid_ckpt_dirs:
                        latest_ckpt_dir = max(valid_ckpt_dirs, key=extract_step_dir)
                        print(f"Found latest valid accelerator checkpoint: {latest_ckpt_dir}")
                    else:
                        print("No valid accelerator checkpoints found (missing trainer_state.json). Starting fresh.")
                else:
                    print("No accelerator checkpoint directories found. Starting fresh.")
            else:
                print(f"No checkpoint directory found at {ckpt_dir}. Starting fresh.")

        path_broadcast_list = [latest_ckpt_dir]
        broadcast_object_list(path_broadcast_list, from_process=0)
        latest_ckpt_dir = path_broadcast_list[0]
        self.accelerator.wait_for_everyone()

        if latest_ckpt_dir is not None:
            try:
                if self.accelerator.is_main_process:
                    print(f"[{self.accelerator.process_index}] Attempting to resume from {latest_ckpt_dir}...")

                self.accelerator.load_state(latest_ckpt_dir)

                # w_path = os.path.join(latest_ckpt_dir, "model.safetensors")
                # if os.path.exists(w_path):
                #     from safetensors.torch import load_file
                #     sd = load_file(w_path)
                # else:
                #     sd = torch.load(os.path.join(latest_ckpt_dir, "pytorch_model.bin"), map_location="cpu")
                # model = self.accelerator.unwrap_model(self.model)
                # curr_sd = model.state_dict()
                # sd = {k: v for k, v in sd.items() if k in curr_sd and v.shape == curr_sd[k].shape}
                # model.load_state_dict(sd, strict=False)
                # print(f"Force loaded weights! Filtered {len(curr_sd) - len(sd)} mismatched keys (Policy reset)")
                # for k, v in curr_sd.items():
                #     if k not in sd:
                #         print(f"  {k} not in old model")

                custom_state_path = os.path.join(latest_ckpt_dir, "trainer_state.json")
                with open(custom_state_path, "r") as f:
                    custom_state = json.load(f)

                self.global_step = custom_state.get("global_step", 0)
                self.epoch = custom_state.get("epoch", 0)
                self.best_metrics_state = custom_state.get("best_metrics_state", {})

                saved_Ts = custom_state.get("shiva_Ts", None)
                saved_Tk = custom_state.get("shiva_Tk", None)

                if saved_Ts is not None and saved_Tk is not None:
                    t_val_Ts = torch.tensor(saved_Ts, device=self.device, dtype=self.dtype)
                    t_val_Tk = torch.tensor(saved_Tk, device=self.device, dtype=self.dtype)
                    for hook in HookManager.get_all_hooks().values():
                        if hasattr(hook, "prune_expert") and hook.prune_expert:
                            hook.prune_expert.Ts = t_val_Ts
                            hook.prune_expert.Tk = t_val_Tk
                        # Also update context pruner
                        if hasattr(hook, "context_pruner") and hook.context_pruner:
                            if hasattr(hook.context_pruner, "Ts"): hook.context_pruner.Ts = t_val_Ts.float()
                            if hasattr(hook.context_pruner, "Tk"): hook.context_pruner.Tk = t_val_Tk.float()
                        break

                resumed = True
                if self.accelerator.is_main_process:
                    print(f"Resumed successfully from step={self.global_step}, epoch={self.epoch}")

            except Exception as e:
                resumed = False
                if self.accelerator.is_main_process:
                    print(f"Failed to load checkpoint {latest_ckpt_dir}. Error: {e}. Starting fresh.")

        self.accelerator.wait_for_everyone()
        if not resumed and self.accelerator.is_main_process:
            print("No checkpoint found or resume failed. Starting fresh.")

        if resumed:
            steps_per_epoch = len(self.train_dataloader) // self.config.gradient_accumulation_steps
            expected_epoch = self.global_step // steps_per_epoch
            if self.epoch != expected_epoch:
                if self.accelerator.is_main_process:
                    print(f"Adjusting epoch from {self.epoch} to {expected_epoch} based on global_step.")
                self.epoch = expected_epoch

    def _initial_phase(self):
        ## resume hook-only state
        if self.config.hook_only_steps > 0:
            is_in_hookonly = self.global_step < self.scaled_hook_only_steps
            self._set_finetuning_trainable_state(requires_grad=not is_in_hookonly)
        else:
            self._set_finetuning_trainable_state(requires_grad=True)

        ## print parameter count
        self._log_trainable_parameters('Initial')

    def _trigger_phase(self):
        # trigger hook-only completion
        if self.config.hook_only_steps > 0:
            if self.global_step == self.scaled_hook_only_steps:
                self._set_finetuning_trainable_state(requires_grad=True)
                self._log_trainable_parameters('After hook-only')

    def train(self):
        if self.accelerator.is_main_process:
            print("Starting training...")
        global_context.update(encoder_flag=True)

        self._try_resume()

        self._initial_phase()

        self.prev_lora_state = None
        self.prev_finetune_state = None

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if hasattr(unwrapped_model, "enable_adapter_layers"):
            unwrapped_model.enable_adapter_layers()

        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            self.model.train()
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
                total=len(self.train_dataloader),
                dynamic_ncols=True,
                disable=not self.accelerator.is_local_main_process,
            )

            for step, batch in enumerate(progress_bar):
                self._update_learning_rate()
                self._update_temperature()

                if step % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.zero_grad(set_to_none=True)

                total_loss, main_loss, hook_loss, distill_loss_final, distill_loss_mid, timesteps = self._train_step(batch)
                is_update_step = (
                    step + 1
                ) % self.config.gradient_accumulation_steps == 0
                is_last_step = (step + 1) == len(self.train_dataloader)

                if self.accelerator.is_main_process:
                    t_val = timesteps.float().mean().item()

                    if self.model_type in ["flux-dev", "flux-kontext"] and t_val <= 1.0:
                        t_val *= 1000.0
                    
                    self.main_viz_buffer.append((t_val, main_loss.float().item()))
                    self.distill_final_viz_buffer.append((t_val, distill_loss_final.float().item()))
                    self.distill_mid_viz_buffer.append((t_val, distill_loss_mid.float().item()))

                if is_update_step or is_last_step:
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                    if self.accelerator.is_main_process and self.global_step % self.log_trigger == 0:
                        try:
                            policy_grad_stats = shiva_metrics.collect_policy_gradient_stats(
                                self.model, HookManager, self.device
                            )
                            self.logger.log_scalars(policy_grad_stats, step=self.global_step)
                        except Exception as e:
                            print(f"Gradient stats collection failed: {e}")

                    self.optimizer.step()
                    self.global_step += 1

                    self._trigger_phase()

                    gathered_losses = {
                        "total": self.accelerator.gather(total_loss).mean(),
                        "main": self.accelerator.gather(main_loss).mean(),
                        "hook": self.accelerator.gather(hook_loss).mean(),
                        "distill_final": self.accelerator.gather(distill_loss_final).mean(),
                        "distill_mid": self.accelerator.gather(distill_loss_mid).mean(),
                    }
                    self._on_update_step_end(gathered_losses, progress_bar, batch)

            if self.accelerator.is_main_process:
                print(f"Epoch {epoch+1} completed.")

        if self.accelerator.is_main_process:
            print("Training completed. Saving final model...")
            final_path = os.path.join(self.config.output_dir, "final.pt")
            self.save_checkpoint(final_path)
            print(f"Final model saved to {final_path}")

    def save_checkpoint(self, save_path: str):
        if self.accelerator.is_main_process:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            model_state = {}
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            state_dict = unwrapped_model.state_dict()
            mode = self.config.training_mode
            if mode == "lora":
                for k, v in state_dict.items():
                    if "lora" in k or "_hook" in k:
                        model_state[k] = v

            elif mode == "hook_only":
                model_state = {k: v for k, v in state_dict.items() if "_hook" in k}

            elif mode == "finetune":
                model_state = state_dict

            else:
                raise ValueError(f"Unknown training_mode: {mode}")

            hook_info = None
            if self.config.trainable_hook_config:
                hook_class, hook_kwargs = self.config.trainable_hook_config
                hook_info = {
                    "class_name": hook_class.__name__,
                    "config": hook_kwargs
                }

            lora_info = None
            if self.config.lora_config and self.config.lora_config.enable:
                lora_info = asdict(self.config.lora_config)

            global_step = getattr(self, "global_step", 0)

            checkpoint = {
                "model": model_state,
                "training_mode": mode,
                "model_type": self.model_type,
                "hook_info": hook_info,
                "lora_info": lora_info,
                "global_step": global_step,
            }

            torch.save(checkpoint, save_path)
            print(
                f"Saved inference checkpoint to {save_path} "
                f"(mode={mode}, step={global_step})"
            )

    def save_specific_best(self, metric_name: str, metric_value: float, higher_is_better: bool = True):
        if not self.accelerator.is_main_process:
            return

        current_best = self.best_metrics_state.get(metric_name)

        is_new_best = False
        if current_best is None:
            is_new_best = True
        else:
            if higher_is_better:
                is_new_best = metric_value > current_best
            else:
                is_new_best = metric_value < current_best

        if is_new_best:
            self.best_metrics_state[metric_name] = metric_value
            filename = f"best_{metric_name.replace('/', '_').replace(' ', '_')}.pt"
            save_path = os.path.join(self.config.output_dir, filename)

            print(f"New best for [{metric_name}]: {metric_value:.4f}. Saving to {filename}...")
            self.save_checkpoint(save_path)

    def maybe_save_periodically(self):
        save_path_dir = os.path.join(
            self.config.output_dir, f"checkpoint_step_{self.global_step:06d}"
        )

        self.accelerator.save_state(save_path_dir)
        if self.accelerator.is_main_process:
            current_Ts = 0.0
            current_Tk = 0.0
            for hook in HookManager.get_all_hooks().values():
                if hasattr(hook, "prune_expert") and hook.prune_expert:
                    current_Ts = hook.prune_expert.Ts.item()
                    current_Tk = hook.prune_expert.Tk.item()
                    break

            custom_state = {
                "global_step": self.global_step,
                "epoch": self.epoch,
                "best_metrics_state": self.best_metrics_state,
                "shiva_Ts": current_Ts,
                "shiva_Tk": current_Tk,
            }
            try:
                with open(os.path.join(save_path_dir, "trainer_state.json"), "w") as f:
                    json.dump(custom_state, f)
            except Exception as e:
                print(f"Warning: Failed to save trainer_state.json: {e}")

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            print(f"Saved accelerator state to {save_path_dir} (step {self.global_step})")

    def load_from_checkpoint(self, ckpt_path, model_only=True):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        print(f">>> Loading checkpoint from {ckpt_path} (model_only={model_only})...")
        if os.path.isfile(ckpt_path):
            print(f"    -> Detected file input. Loading custom weights...")
            if ckpt_path.endswith(".safetensors"):
                if load_safetensors is None:
                    raise ImportError("Please install safetensors.")
                state_dict = load_safetensors(ckpt_path, device=str(self.device))
            else:
                state_dict = torch.load(ckpt_path, map_location=self.device)

            if "model" in state_dict:
                state_dict = state_dict["model"]

            model_to_load = self.accelerator.unwrap_model(self.model)
            missing, unexpected = model_to_load.load_state_dict(state_dict, strict=False)
            print(f"    -> Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            return

        if os.path.isdir(ckpt_path):
            if not model_only:
                print(f"    -> Detected directory with model_only=False. Resuming full trainer state...")
                self.accelerator.load_state(ckpt_path)
                _resume_custom_trainer_state(self, ckpt_path)
                return

            else:
                print(f"    -> Detected directory with model_only=True. Extracting weights only...")
                potential_files = [
                    "model.safetensors",
                    "pytorch_model.bin",
                    "diffusion_pytorch_model.safetensors"
                ]
                weights_path = None
                for f in potential_files:
                    p = os.path.join(ckpt_path, f)
                    if os.path.exists(p):
                        weights_path = p
                        break
                
                if weights_path is None:
                    raise FileNotFoundError(f"No model weights found in {ckpt_path}. Searched for: {potential_files}")
                
                print(f"    -> Found weights file: {weights_path}")

                if weights_path.endswith(".safetensors"):
                    if load_safetensors is None:
                        raise ImportError("Please install safetensors.")
                    state_dict = load_safetensors(weights_path, device=str(self.device))
                else:
                    state_dict = torch.load(weights_path, map_location=self.device)

                model_to_load = self.accelerator.unwrap_model(self.model)
                missing, unexpected = model_to_load.load_state_dict(state_dict, strict=False)
                
                print(f"    -> Accelerator weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
                return

    def _resume_custom_trainer_state(self, ckpt_dir):
        state_path = os.path.join(ckpt_dir, "trainer_state.json")
        if not os.path.exists(state_path):
            print(f"    [Warning] 'trainer_state.json' not found in {ckpt_dir}. Skipping custom state restore.")
            return

        with open(state_path, "r") as f:
            custom_state = json.load(f)

        self.global_step = custom_state.get("global_step", 0)
        self.epoch = custom_state.get("epoch", 0)
        self.best_metrics_state = custom_state.get("best_metrics_state", {})

        saved_Ts = custom_state.get("shiva_Ts", None)
        saved_Tk = custom_state.get("shiva_Tk", None)

        if saved_Ts is not None and saved_Tk is not None:
            print(f"    -> Restoring adaptive pruner state (Ts={saved_Ts}, Tk={saved_Tk})...")
            t_val_Ts = torch.tensor(saved_Ts, device=self.device, dtype=self.dtype)
            t_val_Tk = torch.tensor(saved_Tk, device=self.device, dtype=self.dtype)
            
            for hook in HookManager.get_all_hooks().values():
                # Image Pruner
                if hasattr(hook, "prune_expert") and hook.prune_expert:
                    hook.prune_expert.Ts.copy_(t_val_Ts)
                    hook.prune_expert.Tk.copy_(t_val_Tk)
                
                # Context Pruner
                if hasattr(hook, "context_pruner") and hook.context_pruner:
                    if hasattr(hook.context_pruner, "Ts"): 
                        hook.context_pruner.Ts.copy_(t_val_Ts.float())
                    if hasattr(hook.context_pruner, "Tk"): 
                        hook.context_pruner.Tk.copy_(t_val_Tk.float())
        
        if hasattr(self, "train_dataloader") and self.train_dataloader is not None:
            steps_per_epoch = len(self.train_dataloader) // self.config.gradient_accumulation_steps
            if steps_per_epoch > 0:
                expected_epoch = self.global_step // steps_per_epoch
                if self.epoch != expected_epoch:
                    print(f"    -> Adjusting epoch from {self.epoch} to {expected_epoch} based on global_step.")
                    trainer.epoch = expected_epoch