# 文件: project_name/training/trainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_scheduler
from peft import LoraConfig as PeftLoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
import os
import random
import numpy as np

# 假设您的 project_name 路径已在 PYTHONPATH 中
from project_name.hook import HookManager, BaseDiTBlockHook
from project_name.utils import get_pipeline # 假设 get_pipeline 在 utils.py 中
from project_name.training.config import TrainerConfig, LoraConfig
from project_name.training.dataset import ImageTextDataset, collate_fn

# 确保已安装: pip install diffusers transformers accelerate peft
# (accelerate 用于混合精度, peft 用于 LoRA)

class DiffusionModelTrainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._set_seed(config.seed)

        # 设置混合精度
        self.dtype = torch.float32
        if config.mixed_precision == "fp16":
            self.dtype = torch.float16
        elif config.mixed_precision == "bf16":
            self.dtype = torch.bfloat16

        # 1. 加载模型和预处理器
        # (假设 get_pipeline 返回 (pipe, model_attrs))
        self.pipe, self.model_attrs = get_pipeline(config.model_id)
        
        # 提取核心组件 (适配多 text_encoder)
        self.text_encoder = self.pipe.text_encoder.to(self.device)
        self.text_encoder_2 = getattr(self.pipe, "text_encoder_2", None)
        if self.text_encoder_2:
            self.text_encoder_2 = self.text_encoder_2.to(self.device)
            
        self.vae = self.pipe.vae.to(self.device)
        self.tokenizer = self.pipe.tokenizer
        self.tokenizer_2 = getattr(self.pipe, "tokenizer_2", None)
        self.image_processor = self.pipe.image_processor
        self.scheduler = self.pipe.scheduler
        
        # 'model' 指的是主要的 DiT/U-Net
        self.model = getattr(self.pipe, self.model_attrs['model']).to(self.device)

        print(f"成功加载模型 {config.model_id}。DiT 模型: {self.model.__class__.__name__}")

        # 2. 注入可训练的 Hook
        if config.trainable_hook_config:
            hook_class, hook_kwargs = config.trainable_hook_config
            HookManager.configure_hooks(
                model_type=config.model_type,
                default=(hook_class, hook_kwargs),
            )
            # 注入 hook，hook 会被注册到 self.model (DiT) 的 block 中
            HookManager.inject_hooks(
                model_type=config.model_type, 
                pipe=self.pipe, 
                model_attrs=self.model_attrs
            )
            print(f"已成功注入可训练 Hook: {hook_class.__name__}")
        
        # 3. 设置模型参数和优化器 (核心逻辑)
        self.optimizer = self._setup_model_and_optimizer()

        # 4. 准备数据集
        self.train_dataset = ImageTextDataset(
            metadata_file=config.metadata_file,
            image_dir=config.image_dir,
            tokenizer=self.tokenizer,
            tokenizer_2=self.tokenizer_2, # (传入)
            image_processor=self.image_processor,
            model_type=config.model_type  # (传入)
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=config.train_batch_size, # (使用 micro-batch size)
            shuffle=True,
            collate_fn=collate_fn # (使用自定义 collate_fn)
        )
        
        # 5. 准备 LR Scheduler (基于 optimizer steps)
        num_update_steps_per_epoch = max(1, len(self.train_dataloader) // config.gradient_accumulation_steps)
        self.num_training_steps = num_update_steps_per_epoch * config.num_epochs
        self.lr_scheduler = self._setup_scheduler()
        
        print(f"--- 训练配置 ---")
        print(f"  模式: {config.training_mode}")
        print(f"  Micro Batch Size: {config.train_batch_size}")
        print(f"  梯度累计: {config.gradient_accumulation_steps}")
        print(f"  有效 Batch Size: {config.train_batch_size * config.gradient_accumulation_steps}")
        print(f"  总优化步数: {self.num_training_steps}")
        print(f"------------------")


    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _setup_model_and_optimizer(self):
        """
        根据 training_mode 配置需要训练的参数 (DiT, LoRA, Hooks)
        并冻结 VAE 和 Text Encoders
        """
        
        # (关键) 冻结 VAE 和 Text Encoders
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        if self.text_encoder_2:
            self.text_encoder_2.requires_grad_(False)
            
        params_to_train = []
        
        if self.config.training_mode == "hook_only":
            print("训练模式: hook_only (仅训练 Hook 参数)")
            self.model.requires_grad_(False) # 冻结 DiT
            # 仅解冻 hook 参数
            for module in self.model.modules():
                if isinstance(module, BaseDiTBlockHook): # (依赖 project_name.hook)
                    module.requires_grad_(True)
                    params_to_train.extend(module.parameters())
            
            if not params_to_train:
                raise ValueError("在 'hook_only' 模式下未找到可训练的 Hook 参数。")
                
        elif self.config.training_mode == "finetune":
            print("训练模式: finetune (训练 Hook + 完整 DiT 模型)")
            self.model.requires_grad_(True) # 解冻 DiT
            params_to_train = list(self.model.parameters())
            # (Hook 参数已包含在 self.model.parameters() 中)

        elif self.config.training_mode == "lora":
            print("训练模式: lora (训练 Hook + DiT LoRA 参数)")
            if not self.config.lora_config.enable:
                raise ValueError("LoraConfig 未启用，无法使用 'lora' 模式。")
            
            # 1. 冻结所有
            self.model.requires_grad_(False)
            
            # 2. 应用 LoRA
            lora_conf = self.config.lora_config
            peft_config = PeftLoraConfig(
                r=lora_conf.rank,
                lora_alpha=lora_conf.lora_alpha,
                target_modules=lora_conf.target_modules,
                lora_dropout=lora_conf.lora_dropout,
                bias="none",
            )
            self.model = get_peft_model(self.model, peft_config)
            print("已应用 LoRA 到 DiT 模型。")
            self.model.print_trainable_parameters()
            
            # (peft 会自动返回可训练的 LoRA 参数)
            params_to_train.extend(self.model.parameters()) 

            # 3. (关键) 再次解冻 Hook 参数
            for module in self.model.modules():
                if isinstance(module, BaseDiTBlockHook):
                    module.requires_grad_(True)
                    params_to_train.extend(module.parameters())
        
        else:
            raise ValueError(f"未知的 training_mode: {self.config.training_mode}")

        # 设置优化器
        if self.config.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                params_to_train, 
                lr=self.config.learning_rate, 
                weight_decay=self.config.weight_decay
            )
        else:
            # TODO: 添加更多优化器
            optimizer = torch.optim.Adam(params_to_train, lr=self.config.learning_rate)
            
        return optimizer

    def _setup_scheduler(self):
        scheduler = get_scheduler(
            self.config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=0, # (可配置)
            num_training_steps=self.num_training_steps,
        )
        return scheduler

    @torch.no_grad() # (关键) VAE 和 Text Encoder 在 no_grad() 下运行
    def _prepare_model_inputs(self, batch):
        """
        在冻结的模块(VAE, TextEncoders)上执行前向传播，以节省显存。
        适配 FLUX / PixArt / SD 训练流程。
        """
        
        # 1. VAE 编码
        pixel_values = batch["pixel_values"].to(self.device, dtype=self.vae.dtype)
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # 2. 采样 Noise 和 Timesteps
        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, 
            (batch_size,), device=self.device
        ).long()

        # 3. 添加噪声 (t 步)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # 4. 准备文本条件 (模型特定)
        model_kwargs = {}
        
        if self.config.model_type == "flux-dev":
            # FLUX 使用 T5 (text_encoder_2) 和 CLIP (text_encoder)
            input_ids_2 = batch["input_ids_2"].to(self.device)
            prompt_embeds = self.text_encoder_2(input_ids_2)[0]
            
            input_ids_1 = batch["input_ids_1"].to(self.device)
            pooled_prompt_embeds = self.text_encoder(input_ids_1, output_hidden_states=True)[1]
            
            model_kwargs = {
                "encoder_hidden_states": prompt_embeds.to(self.dtype),
                "pooled_projections": pooled_prompt_embeds.to(self.dtype)
            }

        elif self.config.model_type == "pixart-sigma":
            # PixArt-Sigma 使用 T5 和 added_cond_kwargs
            input_ids = batch["input_ids"].to(self.device)
            prompt_embeds = self.text_encoder(input_ids)[0]
            
            added_conds = batch.get("added_cond_kwargs", {})
            added_conds_processed = {
                "resolution": added_conds.get("resolution").to(self.device),
                "crop_coords": added_conds.get("crop_coords").to(self.device)
            }
            
            model_kwargs = {
                "encoder_hidden_states": prompt_embeds.to(self.dtype),
                "added_cond_kwargs": added_conds_processed
            }
            
        else:
            # 默认 (e.g., Stable Diffusion)
            input_ids = batch["input_ids"].to(self.device)
            prompt_embeds = self.text_encoder(input_ids)[0]
            model_kwargs = {
                "encoder_hidden_states": prompt_embeds.to(self.dtype)
            }

        # 返回 DiT 模型需要的所有输入
        return noisy_latents, timesteps, noise, model_kwargs


    def _compute_loss(self, batch):
        """计算标准的 diffusion 损失 (noise prediction loss)"""
        
        # 1. 准备输入 (VAE, Text Encoders) 
        #    此步骤在 @torch.no_grad() 辅助函数中完成
        noisy_latents, timesteps, noise_target, model_kwargs = self._prepare_model_inputs(batch)
        
        # 2. 运行 DiT/U-Net (此步骤计算梯度)
        #    (使用 autocast 进行混合精度训练)
        with torch.cuda.amp.autocast(enabled=self.config.mixed_precision != "no"):
            noise_pred = self.model(
                hidden_states=noisy_latents.to(self.dtype), # 确保 dtype 匹配
                timestep=timesteps, 
                **model_kwargs # 传入 FLUX 或 PixArt 需要的特定参数
            ).sample

        # 3. 计算主损失 (MSE)
        #    (使用 .float() 计算损失以提高稳定性)
        main_loss = F.mse_loss(noise_pred.float(), noise_target.float(), reduction="mean")
        
        # 4. (可扩展) 计算 Hook 自定义损失
        hook_loss = torch.tensor(0.0, device=self.device)
        custom_losses = []
        for module in self.model.modules():
            if isinstance(module, BaseDiTBlockHook) and hasattr(module, "calculate_custom_loss"):
                # (您需要在您的 Hook (如 DiffCR) 中实现 "calculate_custom_loss" 方法)
                custom_losses.append(module.calculate_custom_loss()) 
        
        if custom_losses:
            hook_loss = torch.stack(custom_losses).mean()

        # 5. 计算总损失
        total_loss = (self.config.main_loss_weight * main_loss + 
                      self.config.hook_loss_weight * hook_loss)
        
        return total_loss, main_loss, hook_loss

    def train(self):
        """
        主训练循环 (支持梯度累计)
        """
        print("开始训练...")
        global_step = 0
        
        # 设置模式
        self.model.train() # (DiT/U-Net 设为训练模式)
        self.vae.eval()    # (VAE 设为评估模式)
        self.text_encoder.eval() # (Text Encoder 1 设为评估模式)
        if self.text_encoder_2:
            self.text_encoder_2.eval()

        for epoch in range(self.config.num_epochs):
            progress_bar = tqdm(
                self.train_dataloader, 
                desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
                total=len(self.train_dataloader)
            )
            
            # (关键) 在梯度累计循环开始前 zero_grad
            self.optimizer.zero_grad()
            
            for step, batch in enumerate(progress_bar):
                
                # --- 梯度累计核心逻辑 ---
                
                # 1. 计算损失 (forward pass)
                total_loss, main_loss, hook_loss = self._compute_loss(batch)
                
                # 2. 缩放损失 (Scale loss for accumulation)
                loss_to_backward = total_loss / self.config.gradient_accumulation_steps
                
                # 3. 反向传播 (backward pass)
                loss_to_backward.backward()
                
                # 4. 检查是否达到累计步数
                is_update_step = (step + 1) % self.config.gradient_accumulation_steps == 0
                is_last_step = (step + 1) == len(self.train_dataloader)

                if is_update_step or is_last_step:
                    # 5. (可选) 梯度裁剪
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) 
                    
                    # 6. 更新优化器 (optimizer step)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    
                    # 7. 重置梯度
                    self.optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # --- 日志 (仅在 optimizer step 时) ---
                    if global_step % self.config.logging_steps == 0:
                        progress_bar.set_postfix(
                            loss=total_loss.item(), # (显示 micro-batch 的 loss)
                            main_loss=main_loss.item(),
                            hook_loss=hook_loss.item(),
                            lr=self.lr_scheduler.get_last_lr()[0],
                            step=global_step
                        )

                    # --- 保存 (仅在 optimizer step 时) ---
                    if global_step % self.config.save_steps == 0:
                        self.save_model(global_step)

        print("训练完成。")
        self.save_model("final") # 保存最终模型

    def save_model(self, step_name):
        """保存训练好的参数 (LoRA 或 Hook)"""
        output_dir = os.path.join(self.config.output_dir, f"checkpoint-{step_name}")
        os.makedirs(output_dir, exist_ok=True)
        
        save_model = self.model
        
        # 如果是 PEFT (LoRA)，保存 adapter
        if isinstance(save_model, PeftModel):
            save_model.save_pretrained(os.path.join(output_dir, "model_lora"))
            print(f"LoRA adapter 已保存至 {output_dir}/model_lora")
        elif self.config.training_mode == "finetune":
            # 否则，如果是完整微调，保存完整 DiT
            torch.save(save_model.state_dict(), os.path.join(output_dir, "model_dit.pt"))
            print(f"完整 DiT 模型已保存至 {output_dir}/model_dit.pt")

        # 始终保存 Hook 参数 (无论哪种模式)
        hook_params = {}
        for name, module in self.model.named_modules():
            if isinstance(module, BaseDiTBlockHook):
                hook_params[name] = module.state_dict()
        
        if hook_params:
            torch.save(hook_params, os.path.join(output_dir, "hook_params.pt"))
            print(f"Hook 参数已保存至 {output_dir}/hook_params.pt")
        elif self.config.training_mode != "finetune":
             print("警告: 未找到 Hook 参数。")