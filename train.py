import math
import pickle
import random
import time
import torch
from typing import List, Dict
import pandas as pd
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch import optim
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


config = {
    'epochs': 50,
    'learning_rate': 0.005,
    'random_seed': 42,
    'hidden_size': 512,
    'rnn_layers': 2,
    'dropout': 0.1

}

data_path = 'data/english/cat_names.txt'
model_path = 'trained_models/english/cat_names'

if __name__ == "__main__":
    writer = SummaryWriter('runs/exp8')

    random.seed(config['random_seed'])
    names = prepare_data(data_path)

    chars = list(set(''.join(names)))
    chars.append('<SOS>')
    chars.append('<EOS>')

    char_to_ix = {ch: i for i, ch in enumerate(sorted(chars))}
    ix_to_char = {i: ch for i, ch in enumerate(sorted(chars))}

    np.random.seed(config['random_seed'])
    np.random.shuffle(names)

    data_size, vocab_size = len(names), len(char_to_ix)
    print('There are %d of training examples and %d unique characters in your data.' % (data_size, vocab_size))

    model = RNN(input_size=vocab_size,
                hidden_size=config['hidden_size'],
                output_size=vocab_size,
                n_layers=config['rnn_layers'],
                dropout=config['dropout'])
    optimizer = optim.Adam(model.parameters(), config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    # print_every = 10
    all_losses = []
    total_loss = 0
    saves = 0

    start = time.time()

    for i in range(config['epochs']):
        for training_example in names:
            model.zero_grad()
            model.hidden = model.init_hidden()
            training_example_tensor = to_tensor(training_example, char_to_ix)
            input = training_example_tensor[:-1]
            target = training_example_tensor[1:]
            loss = 0

            for j in range(input.size(0)):
                output = model(input[j])
                loss += criterion(output, target[j].view(-1))

            total_loss += loss.data.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        avg_loss = total_loss/len(names)
        print('%s (%d %d%%) %.4f' % (timeSince(start), i, i / config['epochs'] * 100, avg_loss))
        writer.add_scalar('data/avg_loss', avg_loss, i)
        total_loss = 0
    print('model trained')

    torch.save(model, model_path + '/model.pt')
    torch.save(model.state_dict(), model_path + '/model_dicts.pt')
    pickle.dump((char_to_ix, ix_to_char), open(model_path + '/dicts.pickle', 'wb'))

    print('model saved')

