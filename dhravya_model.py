import torch
import torch.nn as nn
import torch.nn.functional as func_nn
import numpy as np
import wandb
from torch.distributions import Categorical
from tqdm.auto import tqdm
from text_data import dataset_text, vocab_size, id_to_text, text_to_id


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
text_type = text_file.split(".")[0]
checkpoint_folder = "dhravya"
model_file = f"rnn_dhravya_{text_type}.pth"

# WandB
wandb_config = {"num_epochs": epochs, "lr": lr}
wandb.login()
run = wandb.init(project="dhravya", name="respaper_rnn_1", config=wandb_config)


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

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn_model.parameters(), lr=lr)


def train_step(): #Training step for each epoch
    hidden_state = None
    training_loss = 0
    n_steps = 0
    data_split = np.random.randint(100)

    while True:
        input_sequence = dataset_text[data_split : data_split + seq_len]
        target_sequence = dataset_text[data_split + 1 : data_split + seq_len + 1]

        output, hidden_state = rnn_model(input_sequence, hidden_state)  # forward pass

        loss = loss_fn(torch.squeeze(output), torch.squeeze(target_sequence)) # calculate the loss
        training_loss += loss.item()

        # Compute Training gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update data split for the next text batch
        data_split += seq_len
        n_steps += 1

        if data_split + seq_len + 1 > data_size:
            break # Break loop when the model has iterated tru the entire dataset

        wandb.log({"loss": training_loss/n_steps}) # Log the loss to weights and biases dashboard

    return training_loss/n_steps


def generate_sample(): # generate sample sequence after each epoch
    
    data_pointer = 0
    hidden_state = None
    random_idx = np.random.randint(data_size - 1)
    test_input_seq = dataset_text[random_idx : random_idx + 1] 
    
    while True:
        output, hidden_state = rnn_model(test_input_seq, hidden_state) # get model predictions
        output = func_nn.softmax(torch.squeeze(output), dim=0) 
        distro = torch.distributions.Categorical(output)
        index = distro.sample()

        print(id_to_text[index.item()], end="") # type: ignore
        
        test_input_seq[0][0] = index.item() 
        data_pointer += 1
        
        if data_pointer > out_seq_len:
            break


def train_model():
    
    for epoch in tqdm(range(epochs)):

        train_loss = train_step()

        print(f"epoch {epoch}")
        print('sample text => ')
        generate_sample() # print genrated sample
        print('\n')
        print("----------------------------------------")

        print(f'epoch {epoch} of {epochs}, loss => {train_loss}')


train_model()
