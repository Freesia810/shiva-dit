from project_name.hook import BaseDiTBlockHook

class DiffCRDitBlockHook(BaseDiTBlockHook, nn.Module):
    def __init__(self, ratio_min=0.2, ratio_max=1.0, ratio_step=-0.1, **kwargs):
        '''
            ratio in (ratio_min, ratio_max]
        '''
        super().__init__(**kwargs)
        self.register_buffer("kept_ratio_candidate", torch.arange(ratio_max, ratio_min, ratio_step).float())
        self.diff_ratio = nn.Parameter(torch.tensor(1.0))

    def find_neighbour_bins(self, kept_ratio):
        differences = torch.abs(self.kept_ratio_candidate - kept_ratio)
        _, indices = torch.topk(differences, 2, largest=False)
        neighbour_bins = self.kept_ratio_candidate[indices]
        return neighbour_bins, indices

    def find_closest_bin(self, kept_ratio):
        differences = torch.abs(self.kept_ratio_candidate - kept_ratio)
        closest_index = torch.argmin(differences)
        return self.kept_ratio_candidate[closest_index]

    def on_register_hook(self, block: nn.Module):
        metadata = get_block_metadata(block)
        self.hidden_router = nn.Linear(metadata['hidden_states_dim'], 1, bias=False)
    
    def hook_pre_norm(
        self, 
        module_name: Literal["self-attn", "cross-attn", "ffn", "parallel"],
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        temb: torch.Tensor = None,
    ):
        if module.name == 'self-attn':
            # dit block's start
            seq_len = hidden_states.shape[1]
            kept_ratio = torch.clamp(self.diff_ratio, 0.1, 1.0)
            if self.training:
                pass
            else:
                with torch.no_grad():
                    kept_ratio = self.find_closest_bin(kept_ratio)
                    capacity = ceil_ste(kept_ratio * seq_len).to(torch.int32)
                    k = torch.min(capacity, torch.tensor(seq_len, device=x.device))
                    token_scores = self.hidden_router(x).squeeze(-1) # (batch, seq_len)
        else:
            return super().hook_pre_norm(module_name, hidden_states, encoder_hidden_states, temb)
    

    def hook_after_module(
        self, 
        module_name: Literal["self-attn", "cross-attn", "ffn"],
        hidden_output: torch.Tensor,
        encoder_hidden_output: torch.Tensor = None,
    ):
        if module.name == 'ffn':
            # dit block's end
            pass
        return super().hook_after_module(module_name, hidden_output, encoder_hidden_output)