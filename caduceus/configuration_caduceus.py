"""Caduceus config for Hugging Face.

"""

from typing import Optional, Union

from transformers import PretrainedConfig


class CaduceusConfig(PretrainedConfig):
    """Config that extends the original MambaConfig with params relevant to bi-directionality and RC equivariance."""
    model_type = "caduceus"

    def __init__(
            self,
            # From original MambaConfig
            d_model: int = 2560,
            n_layer: int = 64,
            vocab_size: int = 50277,
            ssm_cfg: Optional[dict] = None,
            rms_norm: bool = True,
            residual_in_fp32: bool = True,
            fused_add_norm: bool = True,
            pad_vocab_size_multiple: int = 8,

            # Not in original MambaConfig, but default arg in create_block in mamba_ssm repo; used in layer norm
            norm_epsilon: float = 1e-5,

            # Used in init_weights
            initializer_cfg: Optional[dict] = None,

            # Caduceus-specific params
            bidirectional: bool = True,
            bidirectional_strategy: Union[str, None] = "add",
            bidirectional_weight_tie: bool = True,
            rcps: bool = False,
            complement_map: Optional[dict] = None,  # used for RCPSEmbedding / RCPSLMHead

            # RNA cross-layer memory
            use_memory: bool = False,
            memory_writer_mode: str = "bcw",
            memory_share_direction_projection: bool = True,
            memory_single_direction: str = "fwd",
            memory_d_sum: int = 256,
            memory_d_mem: int = 128,
            memory_use_write_score: bool = True,
            memory_num_global_slots: int = 1,
            memory_num_local_slots: int = 8,
            memory_pooling: str = "weighted",
            memory_use_layer_embedding: bool = True,
            memory_use_slot_type_embedding: bool = True,
            memory_use_slot_position_embedding: bool = True,
            memory_n_heads: int = 4,
            memory_attn_dropout: float = 0.0,
            memory_reader_gate_init: float = -4.0,
            memory_share_reader: bool = True,
            memory_write_stride: int = 4,
            memory_read_stride: int = 2,
            memory_max_slots: int = 64,
            memory_replacement: str = "fifo",
            memory_collect_stats: bool = False,
            memory_persist_across_batches: bool = False,
            memory_max_size: Optional[int] = None,

            **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.ssm_cfg = ssm_cfg
        self.rms_norm = rms_norm
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.norm_epsilon = norm_epsilon
        self.initializer_cfg = initializer_cfg
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.bidirectional_weight_tie = bidirectional_weight_tie
        self.rcps = rcps
        self.complement_map = complement_map
        
        if memory_max_size is not None:
            # Backward-compatible alias used by early local checkpoints.
            memory_max_slots = memory_max_size

        if memory_writer_mode not in {"single", "average", "scalar_gate", "bcw"}:
            raise ValueError(f"Unsupported memory_writer_mode: {memory_writer_mode}")
        if memory_single_direction not in {"fwd", "bwd"}:
            raise ValueError("memory_single_direction must be 'fwd' or 'bwd'")
        if memory_pooling not in {"mean", "max", "weighted"}:
            raise ValueError(f"Unsupported memory_pooling: {memory_pooling}")
        if memory_pooling == "weighted" and not memory_use_write_score:
            raise ValueError("weighted memory pooling requires memory_use_write_score=true")
        if memory_pooling != "weighted" and memory_use_write_score:
            raise ValueError(
                "Set memory_use_write_score=false when memory_pooling is mean or max"
            )
        if memory_num_global_slots not in {0, 1}:
            raise ValueError("memory_num_global_slots currently supports 0 or 1")
        if memory_num_local_slots < 0:
            raise ValueError("memory_num_local_slots must be non-negative")
        if memory_num_global_slots + memory_num_local_slots == 0:
            raise ValueError("At least one memory slot must be enabled")
        if memory_write_stride <= 0 or memory_read_stride <= 0:
            raise ValueError("Memory read/write strides must be positive")
        if use_memory and n_layer <= memory_read_stride:
            raise ValueError(
                "The configured model has no layer that can read an earlier memory write"
            )
        if memory_max_slots <= 0:
            raise ValueError("memory_max_slots must be positive")
        if memory_replacement != "fifo":
            raise ValueError("Only FIFO memory replacement is currently supported")
        if memory_d_mem % memory_n_heads != 0:
            raise ValueError("memory_d_mem must be divisible by memory_n_heads")

        self.use_memory = use_memory
        self.memory_writer_mode = memory_writer_mode
        self.memory_share_direction_projection = memory_share_direction_projection
        self.memory_single_direction = memory_single_direction
        self.memory_d_sum = memory_d_sum
        self.memory_d_mem = memory_d_mem
        self.memory_use_write_score = memory_use_write_score
        self.memory_num_global_slots = memory_num_global_slots
        self.memory_num_local_slots = memory_num_local_slots
        self.memory_pooling = memory_pooling
        self.memory_use_layer_embedding = memory_use_layer_embedding
        self.memory_use_slot_type_embedding = memory_use_slot_type_embedding
        self.memory_use_slot_position_embedding = memory_use_slot_position_embedding
        self.memory_n_heads = memory_n_heads
        self.memory_attn_dropout = memory_attn_dropout
        self.memory_reader_gate_init = memory_reader_gate_init
        self.memory_share_reader = memory_share_reader
        self.memory_write_stride = memory_write_stride
        self.memory_read_stride = memory_read_stride
        self.memory_max_slots = memory_max_slots
        self.memory_max_size = memory_max_slots
        self.memory_replacement = memory_replacement
        self.memory_collect_stats = memory_collect_stats
        self.memory_persist_across_batches = memory_persist_across_batches
