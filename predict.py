import pickle

import torch

from train import prepare_data

if __name__ == "__main__":
    number_of_generated_names = 10
    minimal_generated_name_length = 10
    model = torch.load('trained_models/char-rnn-generation.pt')
    model.eval()
    char_to_ix, ix_to_char = pickle.load(open('trained_models/dicts.pickle', 'rb'))

    names = prepare_data('data/polish/forenames.txt')
    generated_names = []

    while len(generated_names) < number_of_generated_names:
        prime_str = '<SOS>'
        prime_input = torch.tensor(char_to_ix[prime_str]).to(dtype=torch.long)

        model.init_hidden()
        _ = model(prime_input)

        input = prime_input
        predicted_char = ''
        word = prime_str
        i = 0
        while predicted_char != '<EOS>':
            output = model(input)
            output_dist = output.data.view(-1).exp()

            top_i = torch.multinomial(output_dist, 1)[0]
            predicted_char = ix_to_char[top_i.item()]
            word += predicted_char
            input = torch.tensor(char_to_ix[predicted_char]).to(dtype=torch.long)
            i += 1

        word = word.replace('<SOS>', '')
        word = word.replace('<EOS>', '')
        if len(word) >= minimal_generated_name_length and word not in names and word not in generated_names:
            print(word)
            generated_names.append(word)
