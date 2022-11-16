import json
import copy
import random
import os
import numpy as np
import torch

def get_IIT_MoNLI_dataset(embed_func, suffix, size):
    dataset = IIT_MoNLIDataset(
        embed_func=embed_func,
        suffix=suffix,
        size=size)
    X_base, y_base, X_sources,  y_IIT, interventions = dataset.create()
    y_base = torch.tensor(y_base)
    y_IIT = torch.tensor(y_IIT)
    interventions = torch.tensor(interventions)
    return X_base, y_base, X_sources,  y_IIT, interventions

def get_NMoNLI_dataset(embed_func, suffix):
    dataset = NMoNLIDataset(
        embed_func=embed_func,
        suffix=suffix)
    X_base, y_base = dataset.create()
    y_base = torch.tensor(y_base)
    return X_base, y_base

def get_PMoNLI_dataset(embed_func):
    dataset = PMoNLIDataset(embed_func=embed_func)
    X_base, y_base = dataset.create()
    y_base = torch.tensor(y_base)
    return X_base, y_base

class NMoNLIDataset:
    ENTAIL_LABEL = 0
    NEUTRAL_LABEL = 1
    CONTRADICTION_LABEL = 2

    def __init__(self, embed_func, suffix):
        self.embed_func = embed_func
        self.suffix = suffix

    def create(self):
        nmonli = []
        with open(os.path.join(f"datasets", f"nmonli_{self.suffix}.jsonl")) as f:
            for line in f.readlines():
                example =json.loads(line)
                if example["gold_label"] == "entailment":
                    base_label = self.ENTAIL_LABEL
                else:
                    base_label = self.NEUTRAL_LABEL
                base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
                nmonli.append((base, base_mask, base_label))

        base, base_mask, y = zip(*nmonli)
        self.base = base
        self.base_mask = base_mask
        self.y = np.array(y)
        return (self.base, self.base_mask), self.y

class PMoNLIDataset:
    ENTAIL_LABEL = 0
    NEUTRAL_LABEL = 1
    CONTRADICTION_LABEL = 2

    def __init__(self, embed_func):
        self.embed_func = embed_func

    def create(self):
        pmonli = []
        with open(os.path.join(f"datasets", f"pmonli.jsonl")) as f:
            for line in f.readlines():
                example =json.loads(line)
                if example["gold_label"] == "entailment":
                    base_label = self.ENTAIL_LABEL
                else:
                    base_label = self.NEUTRAL_LABEL
                base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
                pmonli.append((base, base_mask, base_label))

        base, base_mask, y = zip(*pmonli)
        self.base = base
        self.base_mask = base_mask
        self.y = np.array(y)
        return (self.base, self.base_mask), self.y


class IIT_MoNLIDataset:

    ENTAIL_LABEL = 0
    NEUTRAL_LABEL = 1
    CONTRADICTION_LABEL = 2

    def __init__(self, embed_func, suffix, size):
        self.embed_func = embed_func
        self.suffix = suffix
        self.size = size

    def create(self):
        pmonli_entail = []
        pmonli_neutral = []
        nmonli_entail = []
        nmonli_neutral = []
        with open(os.path.join("datasets", "pmonli.jsonl")) as f:
            for line in f.readlines():
                example = json.loads(line)
                if example["gold_label"] == "entailment":
                    pmonli_entail.append(example)
                else:
                    pmonli_neutral.append(example)

        with open(os.path.join(f"datasets", f"nmonli_{self.suffix}.jsonl")) as f:
            for line in f.readlines():
                example = json.loads(line)
                if example["gold_label"] == "entailment":
                    nmonli_entail.append(example)
                else:
                    nmonli_neutral.append(example)


        data = []

        sent_entail = pmonli_entail + nmonli_entail
        neutral = pmonli_neutral + nmonli_neutral

        word_entail = pmonli_entail + nmonli_neutral
        word_neutral = pmonli_neutral + nmonli_entail

        def get_intervention(base,source):
            SEP_ID = 1012
            print(base)
            print((base == SEP_ID).nonzero(as_tuple=True))
            SEP_IND = int((base == SEP_ID).nonzero(as_tuple=True)[0][0])
            return min([i if int(base[0,SEP_IND +i]) != int(base[0,i]) else base.shape[1] + 42 for i in range(base.shape[1])])

        while True:
            example = random.choice(pmonli_entail)
            example2 = random.choice(word_entail)
            base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
            source, source_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
            base_label = self.ENTAIL_LABEL
            intervention = get_intervention(base,source)
            IIT_label = self.ENTAIL_LABEL
            data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))

            example = random.choice(pmonli_entail)
            example2 = random.choice(word_neutral)
            base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
            source, source_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
            base_label = self.ENTAIL_LABEL
            intervention = get_intervention(base,source)
            IIT_label = self.NEUTRAL_LABEL
            data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))


            example = random.choice(nmonli_entail)
            example2 = random.choice(word_entail)
            base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
            source, source_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
            base_label = self.ENTAIL_LABEL
            intervention = get_intervention(base,source)
            IIT_label = self.NEUTRAL_LABEL
            data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))

            example = random.choice(nmonli_entail)
            example2 = random.choice(word_neutral)
            base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
            source, source_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
            base_label = self.ENTAIL_LABEL
            intervention = get_intervention(base,source)
            IIT_label = self.ENTAIL_LABEL
            data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))

            example = random.choice(pmonli_neutral)
            example2 = random.choice(word_entail)
            base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
            source, source_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
            base_label = self.NEUTRAL_LABEL
            intervention = get_intervention(base,source)
            IIT_label = self.ENTAIL_LABEL
            data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))

            example = random.choice(pmonli_neutral)
            example2 = random.choice(word_neutral)
            base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
            source, source_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
            base_label = self.NEUTRAL_LABEL
            intervention = get_intervention(base,source)
            IIT_label = self.NEUTRAL_LABEL
            data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))


            example = random.choice(nmonli_neutral)
            example2 = random.choice(word_entail)
            base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
            source, source_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
            base_label = self.NEUTRAL_LABEL
            intervention = get_intervention(base,source)
            IIT_label = self.NEUTRAL_LABEL
            data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))

            example = random.choice(nmonli_neutral)
            example2 = random.choice(word_neutral)
            base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
            source, source_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
            base_label = self.NEUTRAL_LABEL
            intervention = get_intervention(base,source)
            IIT_label = self.ENTAIL_LABEL
            data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))

            if len(data) > self.size:
                break
        data.sort(key=lambda x: x[-1])

        base, base_mask, y, source, source_mask, IIT_y, interventions = zip(*data)
        self.base = base
        self.base_mask = base_mask
        self.source = source
        self.source_mask = source_mask
        self.y = np.array(y)
        self.IIT_y = np.array(IIT_y)
        self.interventions = np.array(interventions)
        return (self.base, self.base_mask), self.y, [(self.source,self.source_mask)], self.IIT_y, self.interventions
