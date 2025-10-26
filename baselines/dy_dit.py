import inspect
from typing import Literal, Optional, Tuple
import torch
import torch.nn as nn

from diffusers.models.transformers.transformer_flux import FluxSingleTransformerBlock
from project_name.hook import BaseDiTBlockHook, BaseDelegator, HookManager
from project_name.utils import global_context, get_pipeline, process_quick_start


def _gumbel_sigmoid(logits, tau=1, hard=False, eps=1e-10, training=True, threshold=0.5):
    if training:
        # ~Gumbel(0,1)`
        gumbels1 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        gumbels2 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )

        # print(f'tau = {tau}')
        gumbels1 = (logits + gumbels1 - gumbels2) / tau
        y_soft = gumbels1.sigmoid()
    else:
        y_soft = logits.sigmoid()

    if hard:
        # Straight through.
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).masked_fill(y_soft > threshold, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret

class TokenSelect_baesd_on_x_t_text(nn.Module):
    def __init__(
        self,
        dim_in,
        conditioning_dim,
        is_hard=True,
        threshold=0.5,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear_t = nn.Linear(conditioning_dim, dim_in * 4)
        self.norm = nn.LayerNorm(dim_in, eps=eps, elementwise_affine=elementwise_affine)
        self.full_dim_in = dim_in
        self.router = nn.Linear(dim_in, 1, bias=True)

        self.is_hard = is_hard
        self.threshold = threshold

    def forward(self, x, temb, text, tau=5.0):
        # if self.training or not is_single_block:
        b, l, c = x.shape
        l_text = text.shape[1]

        shift, scale, shift_text, scale_text = self.linear_t(self.silu(temb)).chunk(
            4, dim=1
        )

        # if self.training or not is_single_block:
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        text = self.norm(text) * (1 + scale_text)[:, None, :] + shift_text[:, None, :]

        text = text.mean(dim=1).unsqueeze(1)

        logits = self.router(x + text)

        token_select_mask = _gumbel_sigmoid(
            logits, tau, self.is_hard, threshold=self.threshold, training=self.training
        )

        return token_select_mask

class DynamicDiTBlockHook(BaseDelegator, BaseDiTBlockHook, nn.Module):
    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        if config is None:
            config = {
                "dynamic_width": {
                    "enable": False,
                    "select_threshold": 0.5,
                },
                "dynamic_token": {
                    "enable": True,
                    "select_threshold": 0.5,
                    "tau": 5.0,
                },
            }

        self.config = config
        assert not self.config['dynamic_width']['enable'], 'Dynamic width not supported yet'

        # for Spatial-wise Dynamic Token
        self.token_idx = None  
        self.ff_output = None

    def on_register_hook(self, block: nn.Module):
        hidden_dim, encoder_hidden_dim = global_context.hidden_dim, global_context.encoder_hidden_dim
        if hidden_dim is None or encoder_hidden_dim is None:
            raise ValueError("hidden_dim and encoder_hidden_dim must be set in global_context")

        # Timestep-wise Dynamic Width
        # todo: source code not released yet

        # Spatial-wise Dynamic Token
        if self.config["dynamic_token"]["enable"]:
            self.dynamic_token_selector = TokenSelect_baesd_on_x_t_text(
                dim_in=hidden_dim,
                conditioning_dim=encoder_hidden_dim,
                is_hard=True,
                threshold=self.config["dynamic_token"]["select_threshold"],
            )
        else:
            self.dynamic_token_selector = None

        self.to(next(block.parameters()).device)
        self.dtype = torch.float32

    def hook_pre_norm(
        self, 
        module_name: Literal["self-attn", "cross-attn", "ffn", "parallel"],
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        temb: torch.Tensor = None,
    ):
        if module_name == 'ffn' or module_name == 'parallel':
            if self.config["dynamic_token"]["enable"]:
                token_mask = self.dynamic_token_selector(
                    hidden_states, temb, encoder_hidden_states, tau=self.config["dynamic_token"]["tau"]
                ).reshape(-1, *token_mask.shape[2:]) # (batch * seq_len, 1)
                self.token_idx = token_mask.nonzero()[:, 0]
            else:
                self.token_idx = None
        return super().hook_pre_norm(
            module_name,
            hidden_states,
            encoder_hidden_states,
            image_rotary_emb,
            temb,
        )

    def hook_after_norm(
        self, 
        module_name: Literal["self-attn", "cross-attn", "ffn"],
        norm_hidden_states: torch.Tensor,
        norm_encoder_hidden_states: torch.Tensor = None,
    ):
        if module_name == "ffn" and self.token_idx is not None:
            B, _, C = norm_hidden_states.shape
            self.ff_output = torch.zeros_like(
                norm_hidden_states,
                device=norm_hidden_states.device,
                dtype=norm_hidden_states.dtype,
            )

            norm_hidden_states = norm_hidden_states.reshape(
                -1, *norm_hidden_states.shape[2:]
            )
            norm_hidden_states = norm_hidden_states[self.token_idx, :]
            norm_hidden_states = norm_hidden_states.reshape(B, -1, C)
        return super().hook_after_norm(
            module_name,
            norm_hidden_states,
            norm_encoder_hidden_states,
        )

    def hook_after_module(
        self, 
        module_name: Literal["self-attn", "cross-attn", "ffn"],
        hidden_output: torch.Tensor,
        encoder_hidden_output: torch.Tensor = None,
    ):
        if module_name == 'ffn' and self.token_idx is not None:
            self.ff_output = self.ff_output.reshape(-1, *self.ff_output.shape[2:])
            self.ff_output[self.token_idx, :] = hidden_output
            self.ff_output.reshape(hidden_output.shape[0], global_context.hidden_seq_len, global_context.hidden_dim)
            hidden_output = self.ff_output
            self.ff_output = self.token_idx = None
        return super().hook_after_module(module_name, hidden_output, encoder_hidden_output)

    def delegated_call(self, func, module_self, *args, **kwargs):
        if isinstance(module_self, FluxSingleTransformerBlock) and self.config["dynamic_token"]["enable"]:
            # special forward for FluxSingleTransformerBlock with Dynamic Token
            sig = inspect.signature(func)
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()

            hidden_states = bound.arguments.get("hidden_states", None)
            encoder_hidden_states = bound.arguments.get("encoder_hidden_states", None)
            temb = bound.arguments.get("temb", None)
            joint_attention_kwargs = bound.arguments.get("joint_attention_kwargs", None)

            # simulate forward of FluxSingleTransformerBlock
            text_seq_len = encoder_hidden_states.shape[1]

            ff_output = torch.zeros_like(
                hidden_states,
                device=norm_hidden_states.device,
                dtype=norm_hidden_states.dtype,
            )

            # pre norm
            if module_self._hook is not None:
                hidden_states, encoder_hidden_states, image_rotary_emb = (
                    module_self._hook.hook_pre_norm(
                        "parallel",
                        hidden_states,
                        encoder_hidden_states,
                        image_rotary_emb,
                        temb,
                    )
                )

            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

            residual = hidden_states
            norm_hidden_states, gate = module_self.norm(hidden_states, emb=temb)

            # split
            encoder_normed_hidden_states, normed_hidden_states = norm_hidden_states[:, :text_seq_len], norm_hidden_states[:, text_seq_len:]

            # attn
            joint_attention_kwargs = joint_attention_kwargs or {}
            attn_output = module_self.attn(
                hidden_states=norm_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **joint_attention_kwargs,
            )

            # select attn
            encoder_attn_output, hidden_attn_output = attn_output[:, :text_seq_len], attn_output[:, text_seq_len:]

            # calculate encoder
            encoder_mlp_hidden_states = module_self.act_mlp(
                module_self.proj_mlp(encoder_normed_hidden_states)
            )
            encoder_hidden_states = torch.cat([encoder_attn_output, encoder_mlp_hidden_states], dim=2)
            encoder_hidden_states = module_self.proj_out(encoder_hidden_states)

            # do Spatial-wise Dynamic Token
            B, _, C = normed_hidden_states.shape
            normed_hidden_states = normed_hidden_states.reshape(
                -1, *normed_hidden_states.shape[2:]
            )
            normed_hidden_states = normed_hidden_states[self.token_idx, :]
            normed_hidden_states = normed_hidden_states.reshape(B, -1, C)
            hidden_mlp_hidden_states = module_self.act_mlp(
                module_self.proj_mlp(normed_hidden_states)
            )
            hidden_attn_output = hidden_attn_output.reshape(
                -1, *hidden_attn_output.shape[2:]
            )
            hidden_attn_output = hidden_attn_output[self.token_idx, :]
            hidden_attn_output = hidden_attn_output.reshape(B, -1, C)
            hidden_states = torch.cat(
                [hidden_attn_output, hidden_mlp_hidden_states], dim=2
            )
            hidden_states = module_self.proj_out(hidden_states)
            ff_output = ff_output.reshape(-1, *ff_output.shape[2:])
            ff_output[self.token_idx, :] = hidden_states
            ff_output = ff_output.reshape(B, global_context.hidden_seq_len, C)

            # concat and gate
            hidden_states = torch.cat([encoder_hidden_states, ff_output], dim=1)
            gate = gate.unsqueeze(1)
            hidden_states = gate * hidden_states
            hidden_states = residual + hidden_states
            encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
            if hidden_states.dtype == torch.float16:
                hidden_states = hidden_states.clip(-65504, 65504)
            if encoder_hidden_states.dtype == torch.float16:
                encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

            return encoder_hidden_states, hidden_states
        else:
            return super().delegated_call(func, module_self, *args, **kwargs)


if __name__ == "__main__":
    model_type = "flux-dev"
    pipe, model_attrs = get_pipeline(model_type)

    dydit_config = {
        "dynamic_width": {
            "enable": False,
            "select_threshold": 0.5,
        },
        "dynamic_token": {
            "enable": True,
            "select_threshold": 0,
            "tau": 5.0,
        },
    }

    HookManager.configure_hooks(
        model_type=model_type,
        default=(DynamicDiTBlockHook, dydit_config),
    )

    with HookManager.enabled():
        HookManager.inject_hooks(
            model_type=model_type, pipe=pipe, model_attrs=model_attrs
        )
        res = process_quick_start(pipe)
        res.save(f"{model_type}_diffcr.png")

    print(global_context)
