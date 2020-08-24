import logging
import os
import spacy
from torchtext.datasets import Multi30k
from torchtext.data import Field
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import BertWordPieceTokenizer
from keras_preprocessing.sequence import pad_sequences

logger = logging.getLogger(__name__)

START_TOKEN = '<sos>'
END_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'


def load_spacy(module):
    if module == 'en':
        try:
            nlp = spacy.load('en')
        except OSError:
            import en_core_web_sm
            nlp = en_core_web_sm.load()
    elif module == 'de':
        try:
            nlp = spacy.load('de')
        except OSError:
            import de_core_news_sm
            nlp = de_core_news_sm.load()
    else:
        raise NotImplementedError

    return nlp


def train_tokenizer(tokenizer_class, train_file_path, vocab_size, do_lower):
    special_tokens = [START_TOKEN, END_TOKEN, PAD_TOKEN]
    tokenizer = tokenizer_class(lowercase=do_lower)
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.train([train_file_path], vocab_size=vocab_size)
    return tokenizer


def save_tokenizer(tokenizer, out_path):
    tokenizer.save(out_path)


def read_data(data_path):
    with open(data_path, 'r') as file:
        data = file.readlines()
    return data


def tokenize_examples(tokenizer, examples, max_len, pad_token=2):
    tokens = [tokenizer.encode('<sos> ' + example + '<eos>').ids for example in examples]
    padded = pad_sequences(tokens, maxlen=max_len - 1, padding='post', truncating='post', value=pad_token)
    return padded


class TransformerDataset(Dataset):
    def __init__(self, src, tgt, device='cpu'):
        super().__init__()

        self.src = torch.tensor(src, dtype=torch.long).to(device)
        self.tgt = torch.tensor(tgt, dtype=torch.long).to(device)

        # self.src_mask = torch.tensor(src_mask, dtype=torch.bool).to(device)
        # self.tgt_mask = torch.tensor(tgt_mask, dtype=torch.bool).to(device)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return {
            'src': self.src[idx],
            'tgt': self.tgt[idx],
            # 'src_attn_mask': self.src_mask[idx],
            # 'tgt_attn_mask': self.tgt_mask[idx],
        }


def get_loader(src, tgt, device='cpu', batch_size=8, do_shuffle=True):
    dataset = TransformerDataset(src, tgt, device)
    return DataLoader(dataset, batch_size=batch_size, shuffle=do_shuffle)


def tokenize_text(spacy_instance, text):
    return [tok.text for tok in spacy_instance.tokenizer(text)]


class EnDePreprocessor:
    def __init__(self, device, batch_size, out_src_vocab_file, out_tgt_vocab_file, data_path,
                 src_file_name, tgt_file_name, vocab_size=32000, do_lower=True, max_len=50):
        self.data_path = data_path
        self.src_file_name = src_file_name
        self.tgt_file_name = tgt_file_name
        self.vocab_size = vocab_size
        self.do_lower = do_lower
        self.max_len = max_len

        self.device = device
        self.batch_size = batch_size

        self.out_src_vocab_file = out_src_vocab_file
        self.out_tgt_vocab_file = out_tgt_vocab_file


        self.train_iter = None
        self.val_iter = None
        self.test_iter = None

    def _tokenize_en(self, text):
        return tokenize_text(self.spacy_en, text)

    def _tokenize_de(self, text):
        return tokenize_text(self.spacy_de, text)

    def _load_data(self):
        self.spacy_de = load_spacy('de')
        self.spacy_en = load_spacy('en')
        
        self.src = Field(tokenize=self._tokenize_de,
                         init_token='<sos>',
                         eos_token='<eos>',
                         lower=True,
                         batch_first=True)

        self.tgt = Field(tokenize=self._tokenize_en,
                         init_token='<sos>',
                         eos_token='<eos>',
                         lower=True,
                         batch_first=True)
        logger.info('Start loading data')
        _ = Multi30k.splits(exts=('.de', '.en'),
                            fields=(self.src, self.tgt))

    def _read_data(self):
        logger.info('Start reading data')
        src_train_data = read_data(os.path.join(self.data_path, f'train.{self.src_file_name}'))
        tgt_train_data = read_data(os.path.join(self.data_path, f'train.{self.tgt_file_name}'))

        src_val_data = read_data(os.path.join(self.data_path, f'val.{self.src_file_name}'))
        tgt_val_data = read_data(os.path.join(self.data_path, f'val.{self.tgt_file_name}'))

        src_test_data = read_data(os.path.join(self.data_path, f'test2016.{self.src_file_name}'))
        tgt_test_data = read_data(os.path.join(self.data_path, f'test2016.{self.tgt_file_name}'))

        src_train_data = tokenize_examples(self.src_tokenizer, src_train_data, self.max_len, self.src_pad_idx)
        src_val_data = tokenize_examples(self.src_tokenizer, src_val_data, self.max_len, self.src_pad_idx)
        src_test_data = tokenize_examples(self.src_tokenizer, src_test_data, self.max_len, self.src_pad_idx)

        tgt_train_data = tokenize_examples(self.tgt_tokenizer, tgt_train_data, self.max_len, self.tgt_pad_idx)
        tgt_val_data = tokenize_examples(self.tgt_tokenizer, tgt_val_data, self.max_len, self.tgt_pad_idx)
        tgt_test_data = tokenize_examples(self.tgt_tokenizer, tgt_test_data, self.max_len, self.tgt_pad_idx)

        self.train_iter = get_loader(
            src_train_data,
            tgt_train_data,
            device=self.device,
            batch_size=self.batch_size,
            do_shuffle=True
        )
        self.val_iter = get_loader(
            src_val_data,
            tgt_val_data,
            device=self.device,
            batch_size=self.batch_size,
            do_shuffle=False
        )
        self.test_iter = get_loader(
            src_test_data,
            tgt_test_data,
            device=self.device,
            batch_size=self.batch_size,
            do_shuffle=False
        )


    def _build_tokenizers(self):
        logger.info('Start building tokenizer')
        self.src_tokenizer = train_tokenizer(
            BertWordPieceTokenizer,
            os.path.join(self.data_path, f'train.{self.src_file_name}'),
            self.vocab_size,
            self.do_lower
        )
        self.tgt_tokenizer = train_tokenizer(
            BertWordPieceTokenizer,
            os.path.join(self.data_path, f'train.{self.tgt_file_name}'),
            self.vocab_size,
            self.do_lower
        )
        self.src_pad_idx = self.src_tokenizer.encode('<pad>').ids[0]
        self.tgt_pad_idx = self.tgt_tokenizer.encode('<pad>').ids[0]

        logger.info(f'Saving src tokenizer vocab to {self.out_src_vocab_file}')
        save_tokenizer(self.src_tokenizer, self.out_src_vocab_file)
        logger.info(f'Saving tgt tokenizer vocab to {self.out_tgt_vocab_file}')
        save_tokenizer(self.tgt_tokenizer, self.out_tgt_vocab_file)

    def fit_transform(self):
        self._load_data()
        self._build_tokenizers()
        self._read_data()
