import argparse
import warnings
import logging
import os
import math
import time
import torch
import torch.nn as nn
from lib.preprocess import EnDePreprocessor
from lib.transformer import CustomTransformer, initialize_weights
from lib.loops import train, evaluate, epoch_time
from lib.utils import LOG_FORMAT, TRAIN_ARGS, load_config, str2bool, set_file_logger

logging.basicConfig(level=logging.INFO,
                    format=LOG_FORMAT,
                    )
logger = logging.getLogger(__name__)


def run(model_out_file, out_src_vocab_file, out_tgt_vocab_file, transformer_config, batch_size, lr,
        n_epochs, clip, show_progress):
    data_path = '.data/multi30k'
    preprocessor = EnDePreprocessor(
        transformer_config['device'],
        batch_size,
        out_src_vocab_file,
        out_tgt_vocab_file,
        'WMT'
    )
    preprocessor.fit_transform()

    input_dim = preprocessor.src_tokenizer.get_vocab_size()
    output_dim = preprocessor.tgt_tokenizer.get_vocab_size()

    src_pad_idx = preprocessor.src_pad_idx
    tgt_pad_idx = preprocessor.tgt_pad_idx

    model = CustomTransformer(
        src_pad_idx,
        tgt_pad_idx,
        input_dim,
        output_dim,
        **transformer_config
    ).to(transformer_config['device'])
    model.apply(initialize_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)

    best_valid_loss = float('inf')

    logger.info(f'Start training model for {n_epochs} epochs')
    for epoch in range(n_epochs):

        start_time = time.time()

        train_loss = train(model, preprocessor.train_iter, optimizer, criterion, clip, show_progress)
        valid_loss = evaluate(model, preprocessor.val_iter, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            logger.info('Saving model')
            torch.save(model.state_dict(), model_out_file)

        logger.info(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        logger.info(f'Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        logger.info(f'Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    test_loss = evaluate(model, preprocessor.test_iter, criterion)
    logger.info(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out_model_file', type=str, help='The file to which model will be saved')
    parser.add_argument('out_src_vocab_file', type=str, help='The file to which src vocab will be saved')
    parser.add_argument('out_tgt_vocab_file', type=str, help='The file to which src vocab will be saved')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size', required=False)
    parser.add_argument('--lr', type=float, default=None, help='Learning rate', required=False)
    parser.add_argument('--n_epochs', type=int, default=None, help='Epoch number', required=False)
    parser.add_argument('--clip', type=float, default=None, help='Epoch number', required=False)
    parser.add_argument('--min_freq', type=int, default=None, help='Batch size', required=False)
    parser.add_argument('--run_test', action='store_true', required=False, default=False)
    parser.add_argument('--suppress_deprecated', action='store_true', required=False, default=False)
    parser.add_argument('--show_progress', type=str2bool, required=False)
    parser.add_argument('--log_file', type=str, required=False)
    args = parser.parse_args()

    if args.run_test:
        model_config = load_config('config/test_model_config.yaml')
    else:
        model_config = load_config('config/model_config.yaml')
    train_config = load_config('config/train_config.yaml')

    if args.suppress_deprecated:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
    if args.show_progress is not None:
        train_config['show_progress'] = args.show_progress

    if args.log_file:
        if os.path.exists(args.log_file):
            os.remove(args.log_file)
        set_file_logger(args.log_file)


    for arg in TRAIN_ARGS:
        if getattr(args, arg):
            train_config[arg] = getattr(args, arg)

    run(args.out_model_file, args.out_src_vocab_file, args.out_tgt_vocab_file, model_config, **train_config)