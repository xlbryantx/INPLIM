import torch
import torch.nn as nn
import torch.nn.init as init
from Model.Layers import ContextEncoderLayer, TimeEncoderLayer


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def mask_softmax(attn, mask):
    exp = torch.exp(attn)

    exp = exp.masked_fill(mask == 0, 1e-9)
    sum_mask_exp = torch.sum(exp, dim=1, keepdim=True)
    softmax = exp / sum_mask_exp
    softmax = softmax.masked_fill(mask == 0, 0)
    return softmax


class Doctor(nn.Module):
    def __init__(self, features=173, out_dim=2, emb_dim=128, dropout_context=0.5, dropout_time=0.5, feat_padding_idx=0,
                 time_padding_idx=-1):
        super().__init__()

        self.feat_pad_idx = feat_padding_idx
        self.time_pad_idx = time_padding_idx

        self.input_emb = nn.Embedding(num_embeddings=features+1, embedding_dim=emb_dim, padding_idx=0)
        self.time_emb = nn.Linear(in_features=1, out_features=emb_dim, bias=False)

        self.context_aware_encoder = ContextEncoderLayer(d_model=emb_dim, d_inner=emb_dim, n_head=2, d_k=emb_dim,
                                                         d_v=emb_dim, dropout=dropout_context)
        self.time_aware_encoder = TimeEncoderLayer(d_model=emb_dim, n_head=1, d_k=emb_dim, d_v=emb_dim,
                                                   dropout=dropout_time)

        self.w_alpha = nn.Linear(in_features=emb_dim, out_features=1, bias=False)
        self.w_beta = nn.Linear(in_features=emb_dim, out_features=1, bias=False)
        self.w_time = nn.Linear(in_features=emb_dim, out_features=1, bias=False)
        self.w_fuse = nn.Linear(in_features=emb_dim*2, out_features=emb_dim, bias=False)
        self.w_out = nn.Linear(in_features=emb_dim, out_features=out_dim, bias=True)

        init.xavier_normal_(self.input_emb.weight)
        init.xavier_normal_(self.time_emb.weight)
        init.xavier_normal_(self.w_alpha.weight)
        init.xavier_normal_(self.w_beta.weight)
        init.xavier_normal_(self.w_fuse.weight)
        init.xavier_normal_(self.w_out.weight)

    def forward(self, seq, times):

        context_mask = get_pad_mask(seq, pad_idx=self.feat_pad_idx).clone().detach()
        time_mask = get_pad_mask(times, pad_idx=self.time_pad_idx).clone().detach()

        b, c = seq.shape

        feat_emb = self.input_emb(seq).view((b, c, -1))

        context_mask = context_mask.view(b, 1, c)
        context_hidden, context_self_attn = self.context_aware_encoder(feat_emb, context_mask)
        alpha = mask_softmax(self.w_alpha(context_hidden), context_mask.transpose(1, 2).float())
        context_aware_rep = torch.bmm(alpha.transpose(1, 2), feat_emb).squeeze(1)

        time_emb = self.time_emb(times.unsqueeze(-1))
        time_hidden, time_self_attn = self.time_aware_encoder(feat_emb, time_emb, time_mask)
        beta = mask_softmax(self.w_beta(time_hidden) + self.w_time(time_emb),
                            time_mask.transpose(1, 2).float())
        time_aware_rep = torch.bmm(beta.transpose(1, 2), feat_emb).squeeze(1)

        rep = self.w_fuse(torch.cat((context_aware_rep, time_aware_rep), dim=1))

        output = self.w_out(rep)

        return output, alpha, beta, feat_emb
