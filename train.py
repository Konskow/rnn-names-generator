import math
import pickle
import random
import time
import torch
from typing import List, Dict
import pandas as pd
import numpy as np
import torch.nn as nn
from torch import optim
import comet_ml
from rnn import RNN


def random_training_example(data: List[str]) -> str:
    random_number = random.randint(0, len(data) - 1)
    return data[random_number].lower()


def to_tensor(line: str, char_to_ix: Dict[str, int]) -> torch.Tensor:
    res = np.zeros(len(line) + 2)
    res[0] = char_to_ix['<SOS>']
    for i in range(len(line)):
        res[i + 1] = char_to_ix[line[i]]
    res[-1] = char_to_ix['<EOS>']

    return torch.tensor(res).to(dtype=torch.long)


def prepare_data(filepath: str) -> List[str]:
    df = pd.read_csv(open(filepath, 'r'), header=None)
    df.drop_duplicates(inplace=True)
    names = df[0].values

    return [name.lower() for name in names]


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


if __name__ == "__main__":
    epochs = 100

    random.seed(42)
    names = prepare_data('data/polish/forenames.txt')

    chars = list(set(''.join(names)))
    chars.append('<SOS>')
    chars.append('<EOS>')

    char_to_ix = {ch: i for i, ch in enumerate(sorted(chars))}

    ix_to_char = {i: ch for i, ch in enumerate(sorted(chars))}
    print(ix_to_char)

    np.random.seed(2137)
    np.random.shuffle(names)

    data_size, vocab_size = len(names), len(char_to_ix)
    print('There are %d of training examples and %d unique characters in your data.' % (data_size, vocab_size))

    learning_rate = 0.003
    model = RNN(vocab_size, 256, vocab_size, n_layers=2)
    optimizer = optim.Adam(model.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss()

    n_iters = 10000
    print_every = 10
    all_losses = []
    total_loss = 0
    saves = 0
    experiment = comet_ml.Experiment(api_key='',
                                     project_name='',
                                     workspace='',
                                     disabled=False)
    experiment.log_parameter('learning_rate', learning_rate)
    experiment.log_parameter('n_iters', n_iters)

    start = time.time()

    for i in range(n_iters):
        model.zero_grad()
        model.hidden = model.init_hidden()
        training_example = random_training_example(names)
        training_example_tensor = to_tensor(training_example, char_to_ix)
        input = training_example_tensor[:-1]
        target = training_example_tensor[1:]
        loss = 0

        for j in range(input.size(0)):
            output = model(input[j])
            loss += criterion(output, target[j].view(-1))

        total_loss += loss.data[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if i % print_every == 0:
            experiment.log_metric('total_loss', total_loss, step=i)
            print('%s (%d %d%%) %.4f' % (timeSince(start), i, i / n_iters * 100, total_loss))
            total_loss = 0

    torch.save(model, 'trained_models/char-rnn-generation.pt')
    pickle.dump((char_to_ix, ix_to_char), open('trained_models/dicts.pickle', 'wb'))
    print('done')

