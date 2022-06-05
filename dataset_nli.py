import json
import copy
import random
import os
import numpy as np
import torch

def get_IIT_MoNLI_dataset(embed_func, suffix, size):
    train_dataset = IIT_MoNLIDataset(
        embed_func=embed_func,
        suffix=suffix,
        size=size)
    X_base_train, y_base_train, X_sources_train,  y_IIT_train, interventions = train_dataset.create()
    return X_base_train, y_base_train, X_sources_train,  y_IIT_train, interventions

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

        for example in copy.deepcopy(nmonli):
            for example2 in copy.deepcopy(nmonli):
                base = self.embed_func([example["sentence1"], example["sentence2"]])
                source = self.embed_func([example2["sentence1"], example2["sentence2"]])
                intervention = self.LEXVAR

                base_label = self.NEUTRAL_LABEL
                IIT_label = self.NEUTRAL_LABEL
                data.append((base, base_label, source, IIT_label, intervention))
                if len(data) > int(self.size/4):
                    break

        random.shuffle(pmonli)
        random.shuffle(nmonli)

        for example in copy.deepcopy(pmonli):
            for example2 in copy.deepcopy(nmonli):
                base = self.embed_func([example["sentence1"], example["sentence2"]])
                source = self.embed_func([example2["sentence1"], example2["sentence2"]])
                intervention = self.LEXVAR

                base_label = self.ENTAIL_LABEL
                IIT_label = self.NEUTRAL_LABEL
                data.append((base, base_label, source, IIT_label, intervention))
                if len(data) > int((self.size*2)/4):
                    break

        random.shuffle(pmonli)
        random.shuffle(nmonli)

        for example in copy.deepcopy(nmonli):
            for example2 in copy.deepcopy(pmonli):
                base = self.embed_func([example["sentence1"], example["sentence2"]])
                source = self.embed_func([example2["sentence1"], example2["sentence2"]])
                intervention = self.LEXVAR

                base_label = self.NEUTRAL_LABEL
                IIT_label = self.ENTAIL_LABEL
                data.append((base, base_label, source, IIT_label, intervention))
                if len(data) > int((self.size*3)/4):
                    break

        random.shuffle(pmonli)
        random.shuffle(nmonli)

        for example2 in copy.deepcopy(pmonli):
            for example2 in copy.deepcopy(pmonli):
                base = self.embed_func([example["sentence1"], example["sentence2"]])
                source = self.embed_func([example2["sentence1"], example2["sentence2"]])
                intervention = self.LEXVAR

                base_label = self.ENTAIL_LABEL
                IIT_label = self.ENTAIL_LABEL
                data.append((base, base_label, source, IIT_label, intervention))
                if len(data) > int(self.size):
                    break

        random.shuffle(pmonli)
        random.shuffle(nmonli)

        base, source, y, IIT_y, interventions = zip(*data)
        self.base = base
        self.source = source
        self.y = np.array(y)
        self.IIT_y = np.array(IIT_y)
        self.interventions = np.array(interventions)
        self.sources = list()
        self.sources.append(self.source)
        return self.base, self.y, self.sources, self.IIT_y, self.interventions
