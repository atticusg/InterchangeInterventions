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
    return dataset.create()

def get_NMoNLI_dataset(embed_func, suffix):
    dataset = NMoNLIDataset(
        embed_func=embed_func,
        suffix=suffix)
    return dataset.create()

def get_PMoNLI_dataset(embed_func):
    dataset = PMoNLIDataset(embed_func=embed_func)
    return dataset.create()

class NMoNLIDataset:
    ENTAIL_LABEL = 0
    NEUTRAL_LABEL = 1
    CONTRADICTION_LABEL = 2

    def __init__(self, embed_func, suffix):
        self.embed_func = embed_func
        self.suffix = suffix
        self.size = size

    def create(self):
        nmonli = []
        with open(os.path.join(f"datasets", f"nmonli_{self.suffix}.jsonl")) as f:
            for line in f.readlines():
                example =json.loads(line)
                intervention = self.LEXVAR
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
                intervention = self.LEXVAR
                base_label = self.NEUTRAL_LABEL
                base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
                pmonli.append((base, base_mask, base_label))

        base, base_mask, y = zip(*pmonli)
        self.base = base
        self.base_mask = base_mask
        self.y = np.array(y)
        return (self.base, self.base_mask), self.y


class IIT_MoNLIDataset:

    LEXVAR = 0
    ENTAIL_LABEL = 0
    NEUTRAL_LABEL = 1
    CONTRADICTION_LABEL = 2

    def __init__(self, embed_func, suffix, size):
        self.embed_func = embed_func
        self.suffix = suffix
        self.size = size

    def create(self):
        pmonli = []
        nmonli = []
        with open(os.path.join("datasets", "pmonli.jsonl")) as f:
            for line in f.readlines():
                pmonli.append(json.loads(line))
        with open(os.path.join(f"datasets", f"nmonli_{self.suffix}.jsonl")) as f:
            for line in f.readlines():
                nmonli.append(json.loads(line))


        data = []

        while True:
            example = random.choice(nmonli)
            example2 = random.choice(nmonli)
            base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
            source, source_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
            intervention = self.LEXVAR
            base_label = self.NEUTRAL_LABEL
            IIT_label = self.NEUTRAL_LABEL
            data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))

            example = random.choice(pmonli)
            example2 = random.choice(nmonli)
            base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
            source, source_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
            intervention = self.LEXVAR
            base_label = self.ENTAIL_LABEL
            IIT_label = self.NEUTRAL_LABEL
            data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))

            example = random.choice(nmonli)
            example2 = random.choice(pmonli)
            base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
            source, source_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
            intervention = self.LEXVAR
            base_label = self.NEUTRAL_LABEL
            IIT_label = self.ENTAIL_LABEL
            data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))

            example = random.choice(pmonli)
            example2 = random.choice(pmonli)
            base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
            source, source_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
            intervention = self.LEXVAR
            base_label = self.ENTAIL_LABEL
            IIT_label = self.ENTAIL_LABEL
            data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))
            if len(data) > self.size:
                break


        base, base_mask, y, source, source_mask, IIT_y, interventions = zip(*data)
        self.base = base
        self.base_mask = base_mask
        self.source = source
        self.source_mask = source_mask
        self.y = np.array(y)
        self.IIT_y = np.array(IIT_y)
        self.interventions = np.array(interventions)
        self.sources = list()
        self.sources.append((self.source,self.source_mask))
        return (self.base, self.base_mask), self.y, self.sources, self.IIT_y, self.interventions
