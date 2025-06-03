# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Optional, Union

import torch

from megatron.core import tensor_parallel

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_tensor_model_parallel_group_if_none

# pylint: disable=missing-class-docstring
@dataclass
class ApertusMLPSubmodules:
    linear_fc1: Union[ModuleSpec, type] = None
    activation_func: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None


class ApertusMLP(MegatronModule):
    """
    ApertusMLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.


    Returns an output and a bias to be added to the output.
    If config.add_bias_linear is False, the bias returned is None.

    We use the following notation:
     h: hidden size
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: ApertusMLPSubmodules,
        is_expert: bool = False,
        input_size: Optional[int] = None,
        ffn_hidden_size: int = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config

        self.input_size = input_size if input_size != None else self.config.hidden_size
        ffn_hidden_size = self.config.ffn_hidden_size

        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)

        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.input_size,
            ffn_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name='fc1',
            tp_group=tp_group,
        )

        self.activation_func = build_module(
            submodules.activation_func,
        )

        for param in self.activation_func.parameters():
            #print(f"Param: {param}")
            #param.shared = True
            tensor_parallel.set_tensor_model_parallel_attributes(param, True, 0, 1)


        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.config.ffn_hidden_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name='fc2',
            tp_group=tp_group,
        )

    def forward(self, hidden_states):
        """Perform the forward pass through the MLP block."""
        # [s, b, 4 * h/p]
        intermediate_parallel, _ = self.linear_fc1(hidden_states)

        intermediate_parallel = self.activation_func(intermediate_parallel)

        # [s, b, h]
        output, output_bias = self.linear_fc2(intermediate_parallel)

        return output, output_bias