''' Define the Layers '''
import torch.nn as nn
from Model.SubLayers import MultiHeadContextAttention, MultiHeadTimeAttention


class ContextEncoderLayer(nn.Module):

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(ContextEncoderLayer, self).__init__()
        self.slf_attn = MultiHeadContextAttention(n_head, d_model, d_k, d_v, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        return enc_output, enc_slf_attn


class TimeEncoderLayer(nn.Module):

    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1):
        super(TimeEncoderLayer, self).__init__()
        self.slf_attn = MultiHeadTimeAttention(n_head, d_model, d_k, d_v, dropout=dropout)

    def forward(self, enc_input, time_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, time_input, mask=slf_attn_mask)
        return enc_output, enc_slf_attn

