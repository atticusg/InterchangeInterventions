import json
import copy
import random
import os
import numpy as np
import torch

def sample_k_elements(lst, k):
    if k > len(lst):
        ret_lst = []
        for _ in range(k // len(lst)):
            ret_lst.extend(copy.deepcopy(lst))
        ret_lst.extend(random.sample(lst, k % len(lst)))
        random.shuffle(ret_lst)
        return ret_lst
    else:
        # If the list has more than k elements, sample k elements from the list and return them
        return random.sample(lst, k)

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
        assert self.size % 8 == 0

    def create_factual_pairs(
        self,
        pmonli_entail=[],
        pmonli_neutral=[],
        nmonli_entail=[],
        nmonli_neutral=[]
    ):
        if len(pmonli_entail) == 0:
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
            k = self.size // 4
            pmonli_entail = sample_k_elements(pmonli_entail, k)
            pmonli_neutral = sample_k_elements(pmonli_neutral, k)
            nmonli_entail = sample_k_elements(nmonli_entail, k)
            nmonli_neutral = sample_k_elements(nmonli_neutral, k)
            
        data = []

        def get_intervention(base,source):
            SEP_ID = 1012
            SEP_IND = int((base == SEP_ID).nonzero(as_tuple=True)[1][0])
            return min([i if int(base[0,SEP_IND +i]) != int(base[0,i]) else base.shape[1] + 42 for i in range(1,base.shape[1]-SEP_IND)])
        
        for i in range(len(pmonli_entail)):
            example = pmonli_entail[i]
            base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
            base_label = self.ENTAIL_LABEL
            data.append((base, base_mask, base_label))
            
        for i in range(len(nmonli_entail)):
            example = nmonli_entail[i]
            base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
            base_label = self.ENTAIL_LABEL
            data.append((base, base_mask, base_label))
            
        for i in range(len(pmonli_neutral)):
            example = pmonli_neutral[i]
            base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
            base_label = self.NEUTRAL_LABEL
            data.append((base, base_mask, base_label))
            
        for i in range(len(nmonli_neutral)):
            example = nmonli_neutral[i]
            base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
            base_label = self.NEUTRAL_LABEL
            data.append((base, base_mask, base_label))
            
        data.sort(key=lambda x: x[-1])

        base, base_mask, y = zip(*data)
        self.base = base
        self.base_mask = base_mask
        self.y = np.array(y)
        return (self.base, self.base_mask), self.y
    
    def create_neghyp_V1_V2(
        self,
        pmonli_entail=[],
        pmonli_neutral=[],
        nmonli_entail=[],
        nmonli_neutral=[]
    ):
        if len(pmonli_entail) == 0:
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
            k = self.size // 8
            pmonli_entail = sample_k_elements(pmonli_entail, k)
            pmonli_neutral = sample_k_elements(pmonli_neutral, k)
            nmonli_entail = sample_k_elements(nmonli_entail, k)
            nmonli_neutral = sample_k_elements(nmonli_neutral, k)
            
        pool_dict = {
            "pmonli_entail": pmonli_entail,
            "pmonli_neutral": pmonli_neutral,
            "nmonli_entail": nmonli_entail,
            "nmonli_neutral": nmonli_neutral,
        }
        
        data = []

        word_pos = pmonli_entail + pmonli_neutral # 1
        word_neg = nmonli_neutral + nmonli_entail # 0
        word_entail = pmonli_entail + nmonli_neutral # 1
        word_neutral = pmonli_neutral + nmonli_entail # 0
        
        def get_intervention():
            return 2
        
        for i in range(len(pmonli_entail)):
            example = pmonli_entail[i]
            for _ in range(2):
                pos = random.choice([True, False])
                entail = random.choice([True, False])
                example2 = random.choice(word_pos) if pos else random.choice(word_neg)
                example3 = random.choice(word_entail) if entail else random.choice(word_neutral)
                base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
                base_label = self.ENTAIL_LABEL
                source1, source1_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
                source2, source2_mask = self.embed_func([example3["sentence1"], example3["sentence2"]])
                intervention = get_intervention()
                IIT_label = self.ENTAIL_LABEL if pos == entail else self.NEUTRAL_LABEL
                data.append((
                    base, base_mask, base_label, 
                    source1, source2, source1_mask, source2_mask, 
                    IIT_label, intervention
                ))


        for i in range(len(nmonli_entail)):
            example = nmonli_entail[i]
            for _ in range(2):
                pos = random.choice([True, False])
                entail = random.choice([True, False])
                example2 = random.choice(word_pos) if pos else random.choice(word_neg)
                example3 = random.choice(word_entail) if entail else random.choice(word_neutral)
                base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
                base_label = self.ENTAIL_LABEL
                source1, source1_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
                source2, source2_mask = self.embed_func([example3["sentence1"], example3["sentence2"]])
                intervention = get_intervention()
                IIT_label = self.ENTAIL_LABEL if pos == entail else self.NEUTRAL_LABEL
                data.append((
                    base, base_mask, base_label, 
                    source1, source2, source1_mask, source2_mask, 
                    IIT_label, intervention
                ))
    
        for i in range(len(pmonli_neutral)):
            example = pmonli_neutral[i]
            for _ in range(2):
                pos = random.choice([True, False])
                entail = random.choice([True, False])
                example2 = random.choice(word_pos) if pos else random.choice(word_neg)
                example3 = random.choice(word_entail) if entail else random.choice(word_neutral)
                base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
                base_label = self.NEUTRAL_LABEL
                source1, source1_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
                source2, source2_mask = self.embed_func([example3["sentence1"], example3["sentence2"]])
                intervention = get_intervention()
                IIT_label = self.ENTAIL_LABEL if pos == entail else self.NEUTRAL_LABEL
                data.append((
                    base, base_mask, base_label, 
                    source1, source2, source1_mask, source2_mask, 
                    IIT_label, intervention
                ))

        for i in range(len(nmonli_neutral)):
            example = nmonli_neutral[i]
            for _ in range(2):
                pos = random.choice([True, False])
                entail = random.choice([True, False])
                example2 = random.choice(word_pos) if pos else random.choice(word_neg)
                example3 = random.choice(word_entail) if entail else random.choice(word_neutral)
                base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
                base_label = self.NEUTRAL_LABEL
                source1, source1_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
                source2, source2_mask = self.embed_func([example3["sentence1"], example3["sentence2"]])
                intervention = get_intervention()
                IIT_label = self.ENTAIL_LABEL if pos == entail else self.NEUTRAL_LABEL
                data.append((
                    base, base_mask, base_label, 
                    source1, source2, source1_mask, source2_mask, 
                    IIT_label, intervention
                ))
            
        data.sort(key=lambda x: x[-1])

        base, base_mask, y, source1, source2, source1_mask, source2_mask, IIT_y, interventions = zip(*data)
        self.base = base
        self.base_mask = base_mask
        self.source1 = source1
        self.source1_mask = source1_mask
        self.source2 = source2
        self.source2_mask = source2_mask
        self.y = np.array(y)
        self.IIT_y = np.array(IIT_y)
        self.interventions = np.array(interventions)
        return (self.base, self.base_mask), self.y, \
            [(self.source1,self.source1_mask), (self.source2,self.source2_mask)], \
            self.IIT_y, self.interventions
        
    
    def create_neghyp_V1(
        self,
        pmonli_entail=[],
        pmonli_neutral=[],
        nmonli_entail=[],
        nmonli_neutral=[]
    ):
        if len(pmonli_entail) == 0:
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
            k = self.size // 8
            pmonli_entail = sample_k_elements(pmonli_entail, k)
            pmonli_neutral = sample_k_elements(pmonli_neutral, k)
            nmonli_entail = sample_k_elements(nmonli_entail, k)
            nmonli_neutral = sample_k_elements(nmonli_neutral, k)
            
        data = []

        word_pos = pmonli_entail + pmonli_neutral
        word_neg = nmonli_neutral + nmonli_entail
    
        def get_intervention(base,source):
            return 0
        
        for i in range(len(pmonli_entail)):
            example = pmonli_entail[i]
            example2 = random.choice(word_pos)
            base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
            source, source_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
            base_label = self.ENTAIL_LABEL
            intervention = get_intervention(base,source)
            IIT_label = self.ENTAIL_LABEL
            data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))

            example2 = random.choice(word_neg)
            base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
            source, source_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
            base_label = self.ENTAIL_LABEL
            intervention = get_intervention(base,source)
            IIT_label = self.NEUTRAL_LABEL
            data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))
        
        for i in range(len(nmonli_entail)):
            example = nmonli_entail[i]
            example2 = random.choice(word_pos)
            base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
            source, source_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
            base_label = self.ENTAIL_LABEL
            intervention = get_intervention(base,source)
            IIT_label = self.NEUTRAL_LABEL
            data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))

            example2 = random.choice(word_neg)
            base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
            source, source_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
            base_label = self.ENTAIL_LABEL
            intervention = get_intervention(base,source)
            IIT_label = self.ENTAIL_LABEL
            data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))
    
        for i in range(len(pmonli_neutral)):
            example = pmonli_neutral[i]
            example2 = random.choice(word_pos)
            base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
            source, source_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
            base_label = self.NEUTRAL_LABEL
            intervention = get_intervention(base,source)
            IIT_label = self.NEUTRAL_LABEL
            data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))

            example2 = random.choice(word_neg)
            base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
            source, source_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
            base_label = self.NEUTRAL_LABEL
            intervention = get_intervention(base,source)
            IIT_label = self.ENTAIL_LABEL
            data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))

        for i in range(len(nmonli_neutral)):
            example = nmonli_neutral[i]
            example2 = random.choice(word_pos)
            base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
            source, source_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
            base_label = self.NEUTRAL_LABEL
            intervention = get_intervention(base,source)
            IIT_label = self.ENTAIL_LABEL
            data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))

            example2 = random.choice(word_neg)
            base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
            source, source_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
            base_label = self.NEUTRAL_LABEL
            intervention = get_intervention(base,source)
            IIT_label = self.NEUTRAL_LABEL
            data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))
            
        data.sort(key=lambda x: x[-1])

        base, base_mask, y, source, source_mask, IIT_y, interventions = zip(*data)
        self.base = base
        self.base_mask = base_mask
        self.source = source
        self.source_mask = source_mask
        self.y = np.array(y)
        self.IIT_y = np.array(IIT_y)
        self.interventions = np.array(interventions)
        return (self.base, self.base_mask), self.y, [(self.source,self.source_mask), (self.source,self.source_mask)], self.IIT_y, self.interventions
    
    def create_neghyp_V2(
        self,
        pmonli_entail=[],
        pmonli_neutral=[],
        nmonli_entail=[],
        nmonli_neutral=[]
    ):
        if len(pmonli_entail) == 0:
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
            k = self.size // 8
            pmonli_entail = sample_k_elements(pmonli_entail, k)
            pmonli_neutral = sample_k_elements(pmonli_neutral, k)
            nmonli_entail = sample_k_elements(nmonli_entail, k)
            nmonli_neutral = sample_k_elements(nmonli_neutral, k)
            
        data = []

        word_entail = pmonli_entail + nmonli_neutral
        word_neutral = pmonli_neutral + nmonli_entail

        def get_intervention(base,source):
            return 1
        
        for i in range(len(pmonli_entail)):
            example = pmonli_entail[i]
            example2 = random.choice(word_entail)
            base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
            source, source_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
            base_label = self.ENTAIL_LABEL
            intervention = get_intervention(base,source)
            IIT_label = self.ENTAIL_LABEL
            data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))

            example2 = random.choice(word_neutral)
            base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
            source, source_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
            base_label = self.ENTAIL_LABEL
            intervention = get_intervention(base,source)
            IIT_label = self.NEUTRAL_LABEL
            data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))
            
        for i in range(len(nmonli_entail)):
            example = nmonli_entail[i]
            example2 = random.choice(word_entail)
            base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
            source, source_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
            base_label = self.ENTAIL_LABEL
            intervention = get_intervention(base,source)
            IIT_label = self.NEUTRAL_LABEL
            data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))

            example2 = random.choice(word_neutral)
            base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
            source, source_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
            base_label = self.ENTAIL_LABEL
            intervention = get_intervention(base,source)
            IIT_label = self.ENTAIL_LABEL
            data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))
            
        for i in range(len(pmonli_neutral)):
            example = pmonli_neutral[i]
            example2 = random.choice(word_entail)
            base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
            source, source_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
            base_label = self.NEUTRAL_LABEL
            intervention = get_intervention(base,source)
            IIT_label = self.ENTAIL_LABEL
            data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))

            example2 = random.choice(word_neutral)
            base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
            source, source_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
            base_label = self.NEUTRAL_LABEL
            intervention = get_intervention(base,source)
            IIT_label = self.NEUTRAL_LABEL
            data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))

        for i in range(len(nmonli_neutral)):
            example = nmonli_neutral[i]
            example2 = random.choice(word_entail)
            base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
            source, source_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
            base_label = self.NEUTRAL_LABEL
            intervention = get_intervention(base,source)
            IIT_label = self.NEUTRAL_LABEL
            data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))

            example2 = random.choice(word_neutral)
            base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
            source, source_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
            base_label = self.NEUTRAL_LABEL
            intervention = get_intervention(base,source)
            IIT_label = self.ENTAIL_LABEL
            data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))
            
        data.sort(key=lambda x: x[-1])

        base, base_mask, y, source, source_mask, IIT_y, interventions = zip(*data)
        self.base = base
        self.base_mask = base_mask
        self.source = source
        self.source_mask = source_mask
        self.y = np.array(y)
        self.IIT_y = np.array(IIT_y)
        self.interventions = np.array(interventions)
        return (self.base, self.base_mask), self.y, [(self.source,self.source_mask), (self.source,self.source_mask)], self.IIT_y, self.interventions
