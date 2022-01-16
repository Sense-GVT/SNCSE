import json
import en_core_web_sm


special_words = ["am", "is", "was", "are", "were", "can", "could", "will",
                 "would", "shall", "should", "may", "must", "might"]


def convert_to_negation(parser, sentence):

    parsered_sentence = parser(sentence)
    tokens = [str(_) for _ in parsered_sentence]
    deps = [_.dep_ for _ in parsered_sentence]
    tags = [_.tag_ for _ in parsered_sentence]
    lemmas = [_.lemma_ for _ in parsered_sentence]

    if "not" in tokens:
        index = tokens.index("not")
        del tokens[index]
        sentence_negation = " ".join(tokens)
        return sentence_negation

    flag = 0
    for dep in deps:
        if dep == "aux" or dep == "auxpass":
            flag = 1
            break
        if dep == "ROOT":
            flag = 2
            break

    if flag == 1:
        for i, dep in enumerate(deps):
            if dep == "aux" or dep == "auxpass":
                tokens[i] += " not"
                break
    elif flag == 2:
        index = deps.index("ROOT")
        if tokens[index].lower() in special_words:
            tokens[index] += " not"
        elif tags[index] == "VBP":
            tokens[index] = "do not " + lemmas[index]
        elif tags[index] == "VBZ":
            tokens[index] = "does not " + lemmas[index]
        elif tags[index] == "VBD":
            tokens[index] = "did not " + lemmas[index]
        else:
            tokens.insert(0, "Not")
    else:
        tokens.insert(0, "Not")

    sentence_negation = " ".join(tokens)

    return sentence_negation


if __name__ == "__main__":

    parser = en_core_web_sm.load()

    in_file = r"/SNCSE/data/wiki1m_for_simcse.txt"

    out_file = r"/Files/soft_negative_samples.txt"

    f = open(in_file)

    f1 = open(out_file, "w")

    for line in f:
        sentence = line.strip()
        negation = convert_to_negation(parser=parser, sentence=sentence)
        temp = [sentence, negation]
        f1.write(json.dumps(temp) + "\n")

    f.close()
    f1.close()