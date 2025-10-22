from dataclasses import dataclass
from typing import Optional

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

    def pretty(self) -> str:
        """Format the context as a nicely aligned multi-section string."""
        groups = {
            "Basic Info": ["model_name"],
            "Dims": ["hidden_dim", "encoder_hidden_dim"],
            "Hidden Seq": [
                "hidden_seq_len",
                "hidden_seq_height",
                "hidden_seq_width",
                "hidden_seq_frame",
            ],
            "Encoder Seq": [
                "encoder_hidden_seq_len",
                "negetive_encoder_hidden_seq_len",
            ],
            "Edit Extra": [
                "hidden_extra_seq_len",
                "hidden_extra_seq_height",
                "hidden_extra_seq_width",
            ],
        }

        lines = ["PipelineContext {"]
        for section, keys in groups.items():
            lines.append(f"  [{section}]")
            for k in keys:
                v = getattr(self, k, None)
                lines.append(f"    {k:<32}: {v}")
        lines.append("}")
        return "\n".join(lines)

    def __repr__(self):
        return self.pretty()


global_context = PipelineContext()
