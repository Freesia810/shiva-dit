from project_name.hook import BaseDiTBlockHook

class ToMeSDDitBlockHook(BaseDiTBlockHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def hook_pre_norm(
        self, 
        module_name: Literal["self-attn", "cross-attn", "ffn", "parallel"],
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        temb: torch.Tensor = None,
    ):
        return hidden_states, encoder_hidden_states
    
    def hook_after_norm(
        self, 
        module_name: Literal["self-attn", "cross-attn", "ffn"],
        norm_hidden_states: torch.Tensor,
        norm_encoder_hidden_states: torch.Tensor = None,
        temb: torch.Tensor = None,
    ):
        return norm_hidden_states, norm_encoder_hidden_states
    
    def hook_after_module(
        self, 
        module_name: Literal["self-attn", "cross-attn", "ffn"],
        hidden_output: torch.Tensor,
        encoder_hidden_output: torch.Tensor = None,
    ):
        return hidden_output, encoder_hidden_output

    def hook_pre_attn(
        self,
        module_name: Literal["self-attn", "cross-attn"],
        hidden_states_qkv: tuple[
            torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
        ],
        encoder_hidden_states_qkv: tuple[
            torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
        ] = None,
    ):
        return hidden_states_qkv, encoder_hidden_states_qkv

    def hook_after_attn(
        self,
        module_name: Literal["self-attn", "cross-attn"],
        attn_weights: torch.Tensor,
        encoder_attn_weights: torch.Tensor = None,
    ):
        return attn_weights, encoder_attn_weights