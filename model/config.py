# model/config.py
from dataclasses import dataclass, field

@dataclass
class RyuuGPTConfig:
    """
    Configuration for RyuuGPT — optimized for medium-sized code generation models.

    FIXES:
    - use_value_head default changed False (was True — wasted params every run)
    - reasoning_head_kwargs uses dataclass field(default_factory) instead of
      None + __post_init__ hack (safer with dataclass inheritance)
    """

    # Core architecture
    vocab_size: int
    context_size: int  = 768
    n_layer: int       = 12
    n_head: int        = 12
    n_embd: int        = 768

    # Regularization & stability
    dropout: float        = 0.1
    bias: bool            = True
    layer_norm_eps: float = 1e-5

    # Special tokens
    pad_token_id: int = 0
    eos_token_id: int = 1
    bos_token_id: int = 2
    unk_token_id: int = 3

    # ReasoningHead
    use_reasoning_head:    bool  = False
    reasoning_loss_weight: float = 0.1
    reasoning_head_kwargs: dict  = field(default_factory=dict)

    # Modern features
    use_rope:       bool = True   # Rotary position embeddings
    use_rms_norm:   bool = True   # RMSNorm instead of LayerNorm
    use_flash_attn: bool = True   # PyTorch scaled_dot_product_attention

    # FIX: False by default — only enable explicitly for RLHF/reward modelling
    use_value_head: bool = False

    gradient_checkpointing: bool = False

    # Training-related
    dtype: str = "bfloat16"

    # Model identity
    model_type: str = "RyuuGPT-Medium"
    version: str    = "v3.0"