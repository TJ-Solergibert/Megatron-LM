from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType

try:
    from megatron.core.extensions.transformer_engine import (
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TENorm,
        TERowParallelLinear,
    )

except ImportError:
    raise RuntimeError("Apertus models require TransformerEngine")

from .activations import XIELU
from .apertus_mlp import ApertusMLPSubmodules, ApertusMLP


ApertusLayerSpec = ModuleSpec(
    module=TransformerLayer,
    submodules=TransformerLayerSubmodules(
        self_attention=ModuleSpec(
            module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=SelfAttentionSubmodules(
                linear_qkv=TELayerNormColumnParallelLinear,
                core_attention=TEDotProductAttention,
                linear_proj=TERowParallelLinear,
                q_layernorm=TENorm,
                k_layernorm=TENorm,
            ),
        ),
        self_attn_bda=get_bias_dropout_add,
        pre_mlp_layernorm=IdentityOp,
        mlp=ModuleSpec(
            module=ApertusMLP,
            submodules=ApertusMLPSubmodules(
                linear_fc1=TELayerNormColumnParallelLinear,
                activation_func=XIELU,
                linear_fc2=TERowParallelLinear,
            ),
        ),
        mlp_bda=get_bias_dropout_add,
    ),
)