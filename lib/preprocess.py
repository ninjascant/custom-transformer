import logging
import spacy
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

logger = logging.getLogger(__name__)


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


def tokenize_text(spacy_instance, text):
    return [tok.text for tok in spacy_instance.tokenizer(text)]


def train_tokenizer(src, tgt, train_data, min_freq):
    src.build_vocab(train_data, min_freq=min_freq)
    tgt.build_vocab(train_data, min_freq=min_freq)


def get_iterators(train_data, val_data, test_data, batch_size, device):
    return BucketIterator.splits(
        (train_data, val_data, test_data),
        batch_size=batch_size,
        device=device)


def save_vocab(vocab, path):
    with open(path, 'w+') as f:
        for token, index in vocab.stoi.items():
            f.write(f'{index}:\t{token}\n')


def read_vocab(path):
    vocab = dict()
    with open(path, 'r') as f:
        for line in f:
            try:
                index, token = line.split(':\t')
                token = token.replace('\n', '')
                vocab[token] = int(index)
            except ValueError:
                tokens = line.split(':')
                vocab[tokens[0]] = 0
    return vocab


class EnDePreprocessor:
    def __init__(self, device, batch_size, min_freq, out_src_vocab_file, out_tgt_vocab_file):
        self.device = device
        self.batch_size = batch_size
        self.min_freq = min_freq

        self.spacy_de = load_spacy('de')
        self.spacy_en = load_spacy('en')

        self.out_src_vocab_file = out_src_vocab_file
        self.out_tgt_vocab_file = out_tgt_vocab_file

        self.src = None
        self.tgt = None

        self.train_iter = None
        self.val_iter = None
        self.test_iter = None

    def tokenize_en(self, text):
        return tokenize_text(self.spacy_en, text)

    def tokenize_de(self, text):
        return tokenize_text(self.spacy_de, text)

    def fit_transform(self):
        self.src = Field(tokenize=self.tokenize_de,
                    init_token='<sos>',
                    eos_token='<eos>',
                    lower=True,
                    batch_first=True)

        self.tgt = Field(tokenize=self.tokenize_en,
                    init_token='<sos>',
                    eos_token='<eos>',
                    lower=True,
                    batch_first=True)
        logger.info('Start loading data')
        train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                            fields=(self.src, self.tgt))
        logger.info('Start building tokenizer')
        train_tokenizer(self.src, self.tgt, train_data, self.min_freq)
        logger.info('Saving tokenizer vocabs')
        save_vocab(self.src.vocab, self.out_src_vocab_file)
        save_vocab(self.tgt.vocab, self.out_tgt_vocab_file)

        self.train_iter, self.val_iter, self.test_iter = get_iterators(
            train_data,
            valid_data,
            test_data,
            self.batch_size,
            self.device
        )
