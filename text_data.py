import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_file = "respapers_36.txt"

text_data = open(text_file, "rb").read().decode(encoding="utf-8")

print(f"Length of text: {len(text_data)} characters")

text_characters = sorted(list(set(text_data)))
vocab_size = len(text_characters)
text_length = len(text_data)

print(f'the text corpus has {vocab_size} unique characters and is {text_length} characters long..')

text_to_id = {char:id for id, char in enumerate(text_characters)}
id_to_text = {id: char for id, char in enumerate(text_characters)}

dataset_text = list(text_data) 

for idx, char in enumerate(dataset_text):
    dataset_text[idx] = text_to_id[char] # type: ignore

dataset_text = torch.tensor(dataset_text).to(device)
dataset_text = torch.unsqueeze(dataset_text, dim=1)