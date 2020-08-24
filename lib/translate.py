import de_core_news_sm
import torch
from preprocess import read_vocab
from transformer import CustomTransformer


def translate_sentence(sentence, nlp, model, device, src_stoi, tgt_stoi, tgt_itos, max_len=50):
    tgt_init_token = tgt_stoi['<sos>']
    tgt_eos_token = tgt_stoi['<eos>']

    model.eval()

    if isinstance(sentence, str):
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = ['<sos>'] + tokens + ['<eos>']

    src_indexes = [src_stoi[token] for token in tokens]
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

    trg_tokens = [tgt_itos[i] for i in trg_indexes]

    return trg_tokens[1:], attention


class DeTransalator:
    def __init__(self, transformer_config, device, max_len, de_vocab_file, en_vocab_file, model_weights_file):
        self.nlp = de_core_news_sm.load()

        self.de_vocab_stoi = read_vocab(de_vocab_file)
        self.en_vocab_stoi = read_vocab(en_vocab_file)
        self.en_vocab_itos = {self.en_vocab_stoi[key]: key for key in self.en_vocab_stoi.keys()}

        src_pad_idx = self.de_vocab_stoi['<pad>']
        tgt_pad_idx = self.en_vocab_stoi['<pad>']
        input_dim = len(self.de_vocab_stoi)
        output_dim = len(self.en_vocab_stoi)
        self.model = CustomTransformer(src_pad_idx, tgt_pad_idx, input_dim, output_dim, **transformer_config)
        self.model.load_state_dict(torch.load(model_weights_file))

        self.device = device
        self.max_len = max_len

    def predict(self, sentence):
        return translate_sentence(
            sentence,
            self.nlp,
            self.model,
            self.device,
            self.de_vocab_stoi,
            self.en_vocab_stoi,
            self.en_vocab_itos,
            self.max_len
        )