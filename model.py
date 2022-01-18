from typing import Dict, List, Optional, Tuple
import copy
import torch
from torch import nn, Tensor
from torch.nn.init import xavier_normal_
import torch.nn.functional as F
import numpy as np
import random
import json
from numpy.core.numeric import Inf
import math
import torch.nn as nn

class GELU(nn.Module):
    """
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len=20):
        super(PositionalEncoding, self).__init__()

        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])

        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        pad_row = torch.zeros([1, d_model], dtype=torch.float)
        position_encoding = torch.tensor(position_encoding, dtype=torch.float)

        position_encoding = torch.cat((pad_row, position_encoding), dim=0)

        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding, requires_grad=True)

    def forward(self, batch_len, start, seq_len):
        """
        :param batch_len: scalar
        :param seq_len: scalar
        :return: [batch, time, dim]
        """
        input_pos = torch.tensor([list(range(start + 1, start + seq_len + 1)) for _ in range(batch_len)]).cuda()
        return self.position_encoding(input_pos).transpose(0, 1)

class PositionalEncoding1(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, args, dictionary, true_triples=None):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer, Transformer
        except:
            raise ImportError('Transformer module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.ninp = args.embedding_dim
        self.args = args
        self.pos_encoder = PositionalEncoding(self.ninp)
        encoder_layers = nn.TransformerEncoderLayer(d_model=args.embedding_dim, nhead=4, dim_feedforward=args.hidden_size, dropout=args.dropout)
        self.enencoder = nn.TransformerEncoder(encoder_layers, args.num_layers)
        self.ntoken = len(dictionary)
        self.padding_idx = dictionary.pad()
        self.encoder = nn.Embedding(self.ntoken, self.ninp)
        self.fc = torch.nn.Linear(self.ninp, self.ninp)
        self.dictionary = dictionary
        self.glue = GELU()
        self.label_smooth = args.label_smooth

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        xavier_normal_(self.encoder.weight.data)

    def logits(self, source, prev_outputs, **unused):
        bsz, src_len = source.shape
        out_len = prev_outputs.size(1)
        device = source.device
        source = source.transpose(0, 1)
        source = self.encoder(source)
        source += self.pos_encoder(bsz, 0, src_len)
        mask = self._generate_square_subsequent_mask(prev_outputs.size(-1))
        prev_outputs = prev_outputs.transpose(0, 1)
        prev_outputs = self.encoder(prev_outputs)
        prev_outputs += self.pos_encoder(bsz, src_len, out_len)
        if self.args.encoder:
            enmask = torch.zeros(out_len + src_len, out_len + src_len)
            enmask[:, src_len:] = float("-inf")
            enmask[src_len:, src_len:] = mask
            enmask = enmask.to(device)
            output = self.enencoder(torch.cat((source, prev_outputs), dim=0), mask=enmask)[src_len:, :, :].transpose(0, 1)
        else:
            mask = mask.to(device)
            output = self.endecoder(source, prev_outputs, tgt_mask=mask).transpose(0, 1)
        logits = torch.mm(self.glue(self.fc(output)).view(-1, self.ninp), self.encoder.weight.transpose(0, 1)).view(bsz, out_len, -1)
        return logits

    def get_loss(self, source, prev_outputs, target, mask, **unused):
        device = source.device
        bsz = prev_outputs.size(0)
        seq_len = prev_outputs.size(1)
        logits = self.logits(source, prev_outputs)
        # label-smoothing
        lprobs = F.log_softmax(logits, dim=-1)
        loss = -(self.label_smooth * torch.gather(input=lprobs, dim=-1, index=target.unsqueeze(-1)).squeeze() \
            + (1 - self.label_smooth) / (self.ntoken - 1) * lprobs.sum(dim=-1)) * mask
        loss = loss.sum() / mask.sum()
        return loss
