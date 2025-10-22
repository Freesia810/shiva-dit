import torch.nn as nn

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from diffusers.models.attention import JointTransformerBlock, BasicTransformerBlock
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock, FluxSingleTransformerBlock
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock
from diffusers.models.transformers.transformer_wan import WanTransformerBlock

@dataclass
class PipelineContext:
    model_name: Optional[str] = None
    hidden_dim: Optional[int] = None
    encoder_hidden_dim: Optional[int] = None

    hidden_seq_len: Optional[int] = None
    hidden_seq_height: Optional[int] = None
    hidden_seq_width: Optional[int] = None

    encoder_hidden_seq_len: Optional[int] = None
    negetive_encoder_hidden_seq_len: Optional[int] = None

    # for edit task
    hidden_extra_seq_len: Optional[int] = None
    hidden_extra_seq_height: Optional[int] = None
    hidden_extra_seq_width: Optional[int] = None

    # for video generation
    hidden_seq_frame: Optional[int] = None

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError(f"Unknown attribute {k}")
    
    def clear(self):
        for k in self.__dict__.keys():
            setattr(self, k, None)
    
    def __repr__(self):
        return str(self.__dict__)

global_context = PipelineContext()
