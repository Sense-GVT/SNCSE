import json
import os
import numpy as np
import torch
from scipy.spatial.distance import cosine
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm
from scipy.stats import spearmanr
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


def main(ckpt_path, evaluation_file):

    tokenizer = RobertaTokenizer.from_pretrained(r"roberta-large")
    model = RobertaModel.from_pretrained(r"roberta-large").cuda()
    model.eval()

    temp = {"mask_token": tokenizer.mask_token}
    tokenizer.add_special_tokens(temp)

    # File path of STSB development set
    path = r"Files/STSB_dev_set"
    file = os.listdir(path)

    for ckpt in os.listdir(ckpt_path):

        _path = os.path.join(ckpt_path, ckpt)

        if not os.path.isdir(_path):
            continue

        for _ in os.listdir(_path):
            if _ != "pytorch_model.bin":
                os.remove(os.path.join(_path, _))

        _path = os.path.join(_path, "pytorch_model.bin")

        params = torch.load(_path)

        key = [_ for _ in params.keys() if _[8:] in model.state_dict()]
        values = [params[_] for _ in key]

        params = dict(zip([_[8:] for _ in key], values))

        for _ in model.state_dict():
            if _ not in params:
                params[_] = torch.zeros_like(model.state_dict()[_])

        model.load_state_dict(params)

        total_scores = dict()
        _labels = list()
        _scores = list()
        for _file in file:
            if ".input." in _file:
                f = open(os.path.join(path, _file), encoding="utf-8")
                for line in tqdm(f):
                     texts = line.strip().split("\t")
                     texts = [_ + " ." if _.strip()[-1] not in PUNCTUATION else _ for _ in texts]
                     texts[0] = '''This sentence : " ''' + texts[0] + ''' " means <mask> .'''
                     texts[1] = '''This sentence : " ''' + texts[1] + ''' " means <mask> .'''
                     _scores.append(calculate_cosine(tokenizer=tokenizer, model=model, texts=texts))
                f.close()
                _file = _file.replace(".input.", ".gs.")
                f = open(os.path.join(path, _file))
                for line in f:
                    line = line.strip()
                    _labels.append(float(line))
                f.close()

        f = open(evaluation_file, "a")
        _temp = {ckpt: str(spearmanr(_labels, _scores)[0])}
        f.write(str(_temp) + "\n")
        f.close()


if __name__ == "__main__":

    ckpt_path = r"/mnt/lustre/wanghao2/result/final_version/roberta_large"
    evaluation_file = r"/mnt/lustre/wanghao2/result/final_version/roberta_large.json"
    main(ckpt_path=ckpt_path, evaluation_file=evaluation_file)














