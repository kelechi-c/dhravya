import torch, rich, numpy as np
from torch.nn import functional
from rich import traceback
from functools import lru_cache
from dhravya_model import Liquid_Rnn
from text_data import dataset_text, text_to_id, id_to_text

traceback.install(show_locals=True)

input_text = 'transformer models'
model_file = "rnn_dhravya_research.pth"
num_layers = 3
max_len = 300
vocab_size = 348
hidden_size = 512
input_size = vocab_size
random = False
generated_text = ''

# Initialize the model
rnn_model = Liquid_Rnn(input_size, input_size, hidden_size, num_layers)

# Load trained model from file, add cache option
@lru_cache
def load_model(trained_file_path: str):
    rnn_model.load_state_dict(torch.load(trained_file_path))
    num_parameters = sum(p.numel() for p in rnn_model.parameters())
    
    rich.print("[bold green] RNN model(Dhravya_res36) sucessfully loaded.")
    rich.print(f'Number of parameters => {num_parameters}')


def sample():
    load_model(model_file)
    rnn_model.eval()

    seq_counter = 0
    hidden_state = None
    input_seq = []

    if random:
        rand_idx = np.random.randint(200)
        input_seq = dataset_text[rand_idx : rand_idx + 5]
        _, hidden_state = rnn_model(input_seq, hidden_state)
        input_seq = dataset_text[rand_idx + 5 : rand_idx + 6]

    else:
        seed_char = 'gpt'
        next_seq = list(' model')
        chars = list(seed_char)

        for x in chars:
            input_seq.append(text_to_id[x])
            
        _, hidden_state = rnn_model(input_seq, hidden_state)
        
        input_seq = [text_to_id[x] for x in next_seq]
        
        
    print('Generating...')
    
    while True:
        output, hidden_state = rnn_model(input_seq, hidden_state)
        
        output = functional.softmax(torch.squeeze(output, dim=0))
        distro = torch.distributions.Categorical(output)
        output_index = distro.sample().item()
        
        text = id_to_text[output_index] # type: ignore
        
        print(text, end='')
        
        input_seq[0][0] = output_index # type: ignore
        seq_counter += 1
        
        if seq_counter >= max_len:
            break
    
    print('\n ........')
    
sample()

# Run with python3 inference.py 
# add --random=False (random output)
# --max_len=400 (as you want and can generate)