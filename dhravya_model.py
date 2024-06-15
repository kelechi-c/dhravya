import torch
import torch.nn as nn
from text_data import dataset_text, vocab_size


## Hyperparameters
hidden_size = 512
seq_len = 100
num_layers = 3
lr = 0.001
epochs = 50
out_seq_len = 200
load_ckpt = False
data_size = len(dataset_text)
text_file = "researchpapers.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Liquid_Rnn(nn.Module):
    def __init__(self, input_size, out_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(input_size, input_size)
        self.rnn = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers
        )
        self.decoder = nn.Linear(hidden_size, out_size)

    def forward(self, input_seq, hidden_state):
        embeddings = self.embedding(input_seq)
        output, hidden_state = self.rnn(embeddings, hidden_state)
        output = self.decoder(output)

        return output, (hidden_state[0].detach(), hidden_state[1].detach())


rnn_model = Liquid_Rnn(vocab_size, vocab_size, hidden_size, num_layers).to(device)
rnn_model = torch.compile(rnn_model) # use torch.compile 