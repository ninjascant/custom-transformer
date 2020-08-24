import torch
import torch.nn as nn
from lib.model.attention import MultiHeadAttentionLayer, PositionwiseFeedforwardLayer
from lib.model.encoder import Encoder
from lib.model.decoder import Decoder


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
