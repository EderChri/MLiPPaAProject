"""
Transformer implementation.
Taken and adapted from https://github.com/saulam/trajectory_fitting/tree/main
"""
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from constants import DIMENSION


class FittingTransformer(nn.Module):
    """
    Transformer model used for reconstructing trajectories of secondary particles of a particle collision
    """

    def __init__(self,
                 num_encoder_layers: int,
                 d_model: int,
                 n_head: int,
                 input_size: int,
                 output_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 seq_len: int = 15):
        """
        Constructor
        :param num_encoder_layers: number of Transformer encoder layers
        :param d_model: length of the new representation
        :param n_head: number of heads
        :param input_size: size of each item in the input sequence
        :param output_size: size of each item in the output sequence
        :param dim_feedforward: dimension of the feedforward network of the Transformer
        :param dropout: dropout value
        :param seq_len: Optional value to define sequence length as a replacement for the mean
        """
        super(FittingTransformer, self).__init__()
        encoder_layers = TransformerEncoderLayer(d_model=d_model,
                                                 nhead=n_head,
                                                 dim_feedforward=dim_feedforward,
                                                 dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        self.proj_input = nn.Linear(input_size, d_model)
        self.aggregator = nn.Linear(seq_len, 1)
        self.decoder_angle1 = nn.Linear(d_model, output_size)
        self.decoder_angle2 = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self, init_range=0.1) -> None:
        """
        Initialise weights. Inspired by https://github.com/saulam/trajectory_fitting/tree/main
        :param init_range: float, initiale ranges for weights
        :return:
        """
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
        """
        Implements the forward pass of the model.
        :param src: source input tensor
        :param mask: mask to be applied on the attended src. This should be a binary tensor
                    where True values are locations to attend to.
        :param src_padding_mask: source key padding mask. This should be a binary tensor
                                    specifying the locations to ignore.
        :return: If DIMENSION is 2, returns a tensor of track parameters
                If DIMENSION is 3, returns a tuple of tensors each containing a list of track parameters
        """
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
