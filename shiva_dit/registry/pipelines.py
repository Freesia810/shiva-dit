import sys
import os

extern_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "extern", "OmniGen2")
if extern_path not in sys.path:
    sys.path.append(extern_path)

from diffusers.pipelines import (
    StableDiffusion3Pipeline,
    FluxPipeline,
    PixArtSigmaPipeline,
    QwenImagePipeline,
    FluxKontextPipeline,
    QwenImageEditPipeline,
    OmniGenPipeline,
    WanPipeline,
)

try:
    from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
except ImportError:
    print("Warning: OmniGen submodule not found or path incorrect.")
    OmniGen2Pipeline = None

__all__ = [
    "StableDiffusion3Pipeline",
    "FluxPipeline",
    "PixArtSigmaPipeline",
    "QwenImagePipeline",
    "FluxKontextPipeline",
    "QwenImageEditPipeline",
    "OmniGenPipeline",
    "OmniGen2Pipeline",
    "WanPipeline",
    "OmniGen2Pipeline"
]