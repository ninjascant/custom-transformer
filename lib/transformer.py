import math
import torch
import torch.nn as nn
from .model.attention import MultiHeadAttentionLayer, PositionwiseFeedforwardLayer
from .model.encoder import Encoder
from .model.decoder import Decoder


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
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
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class BasicTransformer(nn.Module):
    def __init__(self,
                 src_pad_idx,
                 tgt_pad_idx,
                 src_vocab_size,
                 tgt_vocab_size,
                 num_layers,
                 num_heads,
                 d_model,
                 feed_forward_size,
                 dropout,
                 **kwargs):
        super().__init__()

        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        self.encoder_embed = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embed = nn.Embedding(tgt_vocab_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, feed_forward_size, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, feed_forward_size, dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.out = nn.Linear(feed_forward_size, tgt_vocab_size)

    def forward(self, src, tgt):
        src = src.transpose(1, 0)
        src = self.encoder_embed(src)
        src = self.pos_encoder(src)

        tgt = tgt.transpose(1, 0)
        tgt = self.decoder_embed(tgt)
        tgt = self.pos_encoder(tgt)

        memory = self.encoder(src)
        output = self.decoder(tgt, memory)

        output = self.out(output)
        output = output.transpose(0, 1).contiguous()
        return output


class CustomTransformer(nn.Module):
    def __init__(self,
                 src_pad_idx,
                 trg_pad_idx,
                 input_dim,
                 output_dim,
                 hidden_dim,
                 feed_forward_dim,
                 num_layers,
                 num_heads,
                 enc_dropout,
                 dec_dropout,
                 device
                 ):
        super().__init__()

        self.encoder = Encoder(
            MultiHeadAttentionLayer,
            PositionwiseFeedforwardLayer,
            input_dim,
            hidden_dim,
            num_layers,
            num_heads,
            feed_forward_dim,
            enc_dropout,
            device
        )
        self.decoder = Decoder(
            MultiHeadAttentionLayer,
            PositionwiseFeedforwardLayer,
            output_dim,
            hidden_dim,
            num_layers,
            num_heads,
            feed_forward_dim,
            dec_dropout,
            device
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # src = [batch size, src len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):
        # trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()

        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src, trg):
        # src = [batch size, src len]
        # trg = [batch size, trg len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        enc_src = self.encoder(src, src_mask)

        # enc_src = [batch size, src len, hid dim]

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]

        return output, attention


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
