# model/config.py
from dataclasses import dataclass

@dataclass
class RyuuGPTConfig:
    """
    Configuration for RyuuGPT — optimized for medium-sized code generation models.
    """
    # Core architecture
    vocab_size: int                 # Must match tokenizer vocab size
    context_size: int = 768        # Extended context for code completion
    n_layer: int = 12               # Depth (transformer blocks)
    n_head: int = 12                # Multi-head attention count
    n_embd: int = 768               # Embedding dimension

    # Regularization & stability
    dropout: float = 0.1
    bias: bool = True               # Use bias in linear layers
    layer_norm_eps: float = 1e-5

    # Special tokens
    pad_token_id: int = 0
    eos_token_id: int = 1           # End of sequence
    bos_token_id: int = 2           # Beginning of sequence
    unk_token_id: int = 3           # Unknown token

    # ReasoningHead toggle
    use_reasoning_head: bool = False
    reasoning_head_kwargs: dict = None
    reasoning_loss_weight: float = 0.1

    def __post_init__(self):
        if self.reasoning_head_kwargs is None:
            self.reasoning_head_kwargs = {}  # ← FIX HERE

    # Modern features
    use_rope: bool = True           # Rotary position embeddings (better for long context)
    use_rms_norm: bool = True       # RMSNorm instead of LayerNorm
    use_flash_attn: bool = True     # PyTorch scaled_dot_product_attention
    use_value_head: bool = True     # Enable value head for reward modeling/RLHF
    gradient_checkpointing: bool = False  # Saves VRAM, slower compute

    # Training-related (optional)
    dtype: str = "bfloat16"          # Supports "float32", "bfloat16", "float16"

    # Model identity
    model_type: str = "RyuuGPT-Medium"
    version: str = "v2.0"
