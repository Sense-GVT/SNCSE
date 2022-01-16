import json
import os
import numpy as np
import torch
from scipy.spatial.distance import cosine
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from scipy.stats import spearmanr
import torch.nn as nn
import string

PUNCTUATION = list(string.punctuation)


def calculate_cosine(tokenizer, model, texts):

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    for _ in inputs:
        inputs[_] = inputs[_].cuda()

    temp = inputs["input_ids"]

    # Get the embeddings
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).last_hidden_state.cpu()

    embeddings = embeddings[temp == tokenizer.mask_token_id]

    embeddings = embeddings.tolist()

    score = 1 - cosine(embeddings[0], embeddings[1])

    return score


if __name__ == "__main__":

    tokenizer = BertTokenizer(vocab_file=r"SNCSE-bert-base-uncased/vocab.txt")

    temp = {"mask_token": tokenizer.mask_token}
    tokenizer.add_special_tokens(temp)

    model = BertModel.from_pretrained(r"SNCSE-bert-base-uncased").cuda()

    main_path = r"Files/STS"

    s = []

    for task_name in os.listdir(main_path):

        path = os.path.join(main_path, task_name)

        file = os.listdir(path)

        _labels = list()

        _scores = list()

        for _file in file:
            if "train" in _file or "dev" in _file:
                continue
            if ".input." in _file:
                f = open(os.path.join(path, _file), encoding="utf-8")
                scores = list()
                for line in f:
                    texts = line.strip().split("\t")
                    texts = [_ + " ." if _.strip()[-1] not in PUNCTUATION else _ for _ in texts]
                    texts[0] = '''This sentence : " ''' + texts[0] + ''' " means [MASK] .'''
                    texts[1] = '''This sentence : " ''' + texts[1] + ''' " means [MASK] .'''
                    scores.append(calculate_cosine(tokenizer=tokenizer, model=model, texts=texts))
                f.close()
                _file = _file.replace(".input.", ".gs.")
                labels = list()
                f = open(os.path.join(path, _file))
                for line in f:
                    line = line.strip()
                    try:
                        labels.append(float(line))
                    except:
                        labels.append("NAN")
                f.close()
                index = list()
                for i in range(len(labels)):
                    if labels[i] != "NAN":
                        index.append(i)
                labels = [labels[i] for i in index]
                scores = [scores[i] for i in index]
                _scores.extend(scores)
                _labels.extend(labels)

        print(task_name, spearmanr(_labels, _scores)[0])

        s.append(spearmanr(_labels, _scores)[0])

    print("Avg:", np.mean(s))

