import torch.nn as nn
from diffusers.models.attention import JointTransformerBlock, BasicTransformerBlock
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock, FluxSingleTransformerBlock
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock
from diffusers.models.transformers.transformer_wan import WanTransformerBlock

def get_block_metadata(block: nn.Module):
    if isinstance(block, JointTransformerBlock):
        # SD3 dit block
        assert block.dual_attention is False, "Not implemented for dual attention yet."
        return {
            'hidden_states_dim': block.norm2.normalized_shape[0],
            'encoder_hidden_states_dim': block.norm2_context.normalized_shape[0] if block.norm2_context is not None else None,
            'fusion_type': 'concat',
        }
