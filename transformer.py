"""
Transformer implementation.
Taken from https://github.com/saulam/trajectory_fitting/tree/main
"""
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from constants import DIMENSION


class FittingTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,  # number of Transformer encoder layers
                 d_model: int,  # length of the new representation
                 n_head: int,  # number of heads
                 input_size: int,  # size of each item in the input sequence
                 output_size: int,  # size of each item in the output sequence
                 dim_feedforward: int = 512,  # dimension of the feedforward network of the Transformer
                 dropout: float = 0.1):  # dropout value
        super(FittingTransformer, self).__init__()
        encoder_layers = TransformerEncoderLayer(d_model=d_model,
                                                 nhead=n_head,
                                                 dim_feedforward=dim_feedforward,
                                                 dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        self.proj_input = nn.Linear(input_size, d_model)
        self.decoder_angle1 = nn.Linear(d_model, output_size)
        self.decoder_angle2 = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self, init_range=0.1) -> None:
        # weights initialisation
        self.proj_input.bias.data.zero_()
        self.proj_input.weight.data.uniform_(-init_range, init_range)
        self.decoder_angle1.bias.data.zero_()
        self.decoder_angle1.weight.data.uniform_(-init_range, init_range)
        self.decoder_angle2.bias.data.zero_()
        self.decoder_angle2.weight.data.uniform_(-init_range, init_range)

    def forward(self,
                src: Tensor,
                mask: Tensor,
                src_padding_mask: Tensor):
        # Linear projection of the input
        src_emb = self.proj_input(src)
        # Transformer encoder
        memory = self.transformer_encoder(src=src_emb, mask=mask,
                                          src_key_padding_mask=src_padding_mask)
        memory = torch.mean(memory, dim=0)
        # Dropout
        memory = self.dropout(memory)
        # Linear projection of the output
        if DIMENSION == 2:
            return self.decoder_angle1(memory)
        if DIMENSION == 3:
            output1 = self.decoder_angle1(memory)
            output2 = self.decoder_angle2(memory)
            return output1, output2
