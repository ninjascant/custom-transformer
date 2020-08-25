import logging
import os
import shutil
import requests
import tarfile
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import BertWordPieceTokenizer
from keras_preprocessing.sequence import pad_sequences

logger = logging.getLogger(__name__)

START_TOKEN = '<sos>'
END_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'

WMT_URL = 'https://www.statmt.org/europarl/v7/de-en.tgz'


def train_tokenizer(tokenizer_class, train_file_path, vocab_size, do_lower):
    special_tokens = [START_TOKEN, END_TOKEN, PAD_TOKEN]
    tokenizer = tokenizer_class(lowercase=do_lower)
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.train([train_file_path], vocab_size=vocab_size)
    return tokenizer


def save_tokenizer(tokenizer, out_path):
    tokenizer.save_model(out_path)


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
    def __init__(self, device, batch_size, out_src_vocab_file, out_tgt_vocab_file, dataset_name, vocab_size=32000,
                 do_lower=True, max_len=50):
        self.dataset_name = dataset_name
        self.vocab_size = vocab_size
        self.do_lower = do_lower
        self.max_len = max_len

        self.src_file_name = '.data/data.de'
        self.tgt_file_name = '.data/data.en'

        self.device = device
        self.batch_size = batch_size

        self.out_src_vocab_file = out_src_vocab_file
        self.out_tgt_vocab_file = out_tgt_vocab_file

        self.train_iter = None
        self.val_iter = None
        self.test_iter = None

    def _download_dataset(self, url):
        if os.path.isdir('.data'):
            shutil.rmtree('.data')
        os.mkdir('.data')
        res = requests.get(url, allow_redirects=True)
        with open('.data/data.tgz', 'wb') as file:
            file.write(res.content)
        tar = tarfile.open('.data/data.tgz', "r:gz")
        tar.extractall()
        tar.close()

        os.remove('.data/data.tgz')

    def _load_data(self):
        if not os.path.isfile('.data/data.de'):
            logger.info('Start downloading dataset')
            if self.dataset_name == 'WMT':
                self._download_dataset(WMT_URL)
                os.rename('europarl-v7.de-en.de', self.src_file_name)
                os.rename('europarl-v7.de-en.en', self.tgt_file_name)
        else:
            logger.info('Using cached dataset')

    def _read_data_sample(self, file_name, sample_size):
        with open(file_name, 'r') as file:
            sample = [next(file) for _ in range(sample_size)]
        return sample

    def _get_data_split(self, sample_size, val_size=0.2, test_size=0.2):
        src_sample = self._read_data_sample(self.src_file_name, sample_size)
        tgt_sample = self._read_data_sample(self.tgt_file_name, sample_size)

        self.src_test = src_sample[-20_000:]
        self.tgt_test = src_sample[-20_000:]
        src_sample = src_sample[:-20_000]
        tgt_sample = tgt_sample[:-20_000]

        self.src_train, self.src_val, self.tgt_train, self.tgt_val = train_test_split(src_sample,
                                                                                      tgt_sample,
                                                                                      test_size=0.2)

        with open('.data/src_train.txt', 'w') as outfile:
            outfile.writelines(self.src_train)
        with open('.data/tgt_train.txt', 'w') as outfile:
            outfile.writelines(self.tgt_train)

    def _build_tokenizers(self):
        logger.info('Start building tokenizer')
        self.src_tokenizer = train_tokenizer(
            BertWordPieceTokenizer,
            '.data/src_train.txt',
            self.vocab_size,
            self.do_lower
        )
        self.tgt_tokenizer = train_tokenizer(
            BertWordPieceTokenizer,
            '.data/tgt_train.txt',
            self.vocab_size,
            self.do_lower
        )

        self.src_pad_idx = self.src_tokenizer.encode('<pad>').ids[0]
        self.tgt_pad_idx = self.tgt_tokenizer.encode('<pad>').ids[0]

        logger.info(f'Saving src tokenizer vocab to {self.out_src_vocab_file}')
        save_tokenizer(self.src_tokenizer, self.out_src_vocab_file)
        logger.info(f'Saving tgt tokenizer vocab to {self.out_tgt_vocab_file}')
        save_tokenizer(self.tgt_tokenizer, self.out_tgt_vocab_file)

    def _tokenize_data(self):
        logger.info('Start tokenizing data')
        self.src_train = tokenize_examples(self.src_tokenizer, self.src_train, 50, self.src_pad_idx)
        self.tgt_train = tokenize_examples(self.tgt_tokenizer, self.tgt_train, 50, self.tgt_pad_idx)

        self.src_val = tokenize_examples(self.src_tokenizer, self.src_val, 50, self.src_pad_idx)
        self.tgt_val = tokenize_examples(self.tgt_tokenizer, self.tgt_val, 50, self.tgt_pad_idx)

        self.src_test = tokenize_examples(self.src_tokenizer, self.src_test, 50, self.src_pad_idx)
        self.tgt_test = tokenize_examples(self.tgt_tokenizer, self.tgt_test, 50, self.tgt_pad_idx)

    def _get_loaders(self):
        self.train_iter = get_loader(
            self.src_train,
            self.tgt_train,
            device=self.device,
            batch_size=self.batch_size,
            do_shuffle=True
        )
        self.val_iter = get_loader(
            self.src_val,
            self.tgt_val,
            device=self.device,
            batch_size=self.batch_size,
            do_shuffle=False
        )
        self.test_iter = get_loader(
            self.src_test,
            self.tgt_test,
            device=self.device,
            batch_size=self.batch_size,
            do_shuffle=False
        )

    def fit_transform(self):
        self._load_data()
        self._get_data_split(220_000)
        self._build_tokenizers()
        self._tokenize_data()
        self._get_loaders()
