# Copyright 2023 Natural Synthetics Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn
from diffusers.utils import BaseOutput
from diffusers.models.transformer_2d import Transformer2DModel
from einops import rearrange, repeat
from typing import Dict, Any


@dataclass
class Transformer3DModelOutput(BaseOutput):
    """
        The output of [`Transformer3DModel`].

        Args:
            sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`:
                The hidden states output conditioned on the `encoder_hidden_states` input.
        """

    sample: torch.FloatTensor


class Transformer3DModel(Transformer2DModel):

    def __init__(self, *args, **kwargs):
        super(Transformer3DModel, self).__init__(*args, **kwargs)
        nn.init.zeros_(self.proj_out.weight.data)
        nn.init.zeros_(self.proj_out.bias.data)

    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            class_labels: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            enable_temporal_layers: bool = True,
            positional_embedding: Optional[torch.Tensor] = None,
            return_dict: bool = True,
    ):

        is_video = len(hidden_states.shape) == 5

        if is_video:
            f = hidden_states.shape[2]
            hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
            encoder_hidden_states = repeat(encoder_hidden_states, 'b n c -> (b f) n c', f=f)

        hidden_states = super(Transformer3DModel, self).forward(hidden_states,
                                                                encoder_hidden_states,
                                                                timestep,
                                                                class_labels,
                                                                cross_attention_kwargs,
                                                                attention_mask,
                                                                encoder_attention_mask,
                                                                return_dict=False)[0]

        if is_video:
            hidden_states = rearrange(hidden_states, "(b f) c h w -> b c f h w", f=f)

        if not return_dict:
            return (hidden_states,)

        return Transformer3DModelOutput(sample=hidden_states)