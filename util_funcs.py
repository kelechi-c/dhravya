import re, nltk
from tqdm.auto import tqdm 
from collections import Counter


def tokenize_text(text_input):
    text_tokens = nltk.tokenize.word_tokenize(str(text_input).lower())

    return text_tokens


def preprocess_text(text):
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = tokenize_text(text)
    tokens = [t for t in tokens if t not in nltk.corpus.stopwords.words("english")]
    return tokens 


def create_vocabulary(text_dataset):
    all_tokens = []
    for text in text_dataset:
        tokens = preprocess_text(text)
        all_tokens.extend(tokens)

    vocab_counter = Counter(all_tokens)
    vocab = [word for word, count in vocab_counter.most_common() if count >= 1]
    vocab = ["[pad]", "[start]", "[end]", "[unk]"] + vocab

    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for idx, word in enumerate(vocab)}

    return word_to_idx, idx_to_word