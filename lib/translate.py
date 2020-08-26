import torch
from tokenizers import BertWordPieceTokenizer
from lib.transformer import CustomTransformer


def translate_sentence(sentence, model, device, src_tokenizer, tgt_tokenizer, max_len=50):
    model.eval()

    tgt_init_token = tgt_tokenizer.encode('<sos>').ids[0]
    tgt_eos_token = tgt_tokenizer.encode('<eos>').ids[0]

    src_indexes = src_tokenizer.encode(sentence).ids
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [tgt_init_token]

    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == tgt_eos_token:
            break

    trg_tokens = tgt_tokenizer.decode(trg_indexes)

    return trg_tokens, attention


class DeTransalator:
    def __init__(self, transformer_config, device, max_len, de_vocab_file, en_vocab_file, model_weights_file):
        self.src_tokenizer = self._load_tokenizer(de_vocab_file)
        self.tgt_tokenizer = self._load_tokenizer(en_vocab_file)

        src_pad_idx = self.src_tokenizer.encode('<pad>').ids[0]
        tgt_pad_idx = self.tgt_tokenizer.encode('<pad>').ids[0]
        input_dim = self.src_tokenizer.get_vocab_size() + 3
        output_dim = self.tgt_tokenizer.get_vocab_size() + 3
        self.model = CustomTransformer(src_pad_idx, tgt_pad_idx, input_dim, output_dim, **transformer_config)

        self.model.load_state_dict(torch.load(model_weights_file, map_location=torch.device(device)))

        self.device = device
        self.max_len = max_len

    def _load_tokenizer(self, vocab_file):
        return BertWordPieceTokenizer(vocab_file, lowercase=True)

    def predict(self, sentence):
        return translate_sentence(
            sentence,
            self.model,
            self.device,
            self.src_tokenizer,
            self.tgt_tokenizer,
            self.max_len
        )