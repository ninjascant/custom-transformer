import logging
from tqdm.auto import tqdm
import torch

logger = logging.getLogger(__name__)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def forward_pass(model, src, tgt):
    output, _ = model(src, tgt[:, :-1])
    output_dim = output.shape[-1]
    output = output.contiguous().view(-1, output_dim)
    tgt = tgt[:, 1:].contiguous().view(-1)
    return output, tgt


def train(model, iterator, optimizer, criterion, clip, show_progress):
    model.train()

    epoch_loss = 0
    if show_progress:
        batch_iterator = tqdm(enumerate(iterator), total=len(iterator))
    else:
        batch_iterator = enumerate(iterator)

    for i, batch in batch_iterator:
        src = batch.src
        tgt = batch.trg

        optimizer.zero_grad()
        output, tgt = forward_pass(model, src, tgt)

        loss = criterion(output, tgt)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            tgt = batch.trg

            output, tgt = forward_pass(model, src, tgt)
            loss = criterion(output, tgt)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)
