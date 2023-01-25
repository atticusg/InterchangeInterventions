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
    
    def create_tokenidentity_V1(
        self,
        pmonli_entail=[],
        pmonli_neutral=[],
        nmonli_entail=[],
        nmonli_neutral=[]
    ):
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
        with open(os.path.join(f"datasets", f"nmonli_train.jsonl")) as f:
            for line in f.readlines():
                example = json.loads(line)
                if example["gold_label"] == "entailment":
                    nmonli_entail.append(example)
                else:
                    nmonli_neutral.append(example)

        example_dedup = set([])
        for example in pmonli_entail+pmonli_neutral+nmonli_entail+nmonli_neutral:
            example_dedup.add(example['sentence1']+example['sentence2'])

        for example in pmonli_entail:
            if example['sentence1']+example['sentence2'] in example_dedup:
                continue
            example_dedup.add(example['sentence1']+example['sentence2'])
            new_example = {}
            new_example['sentence1'] = example['sentence2']
            new_example['sentence2'] = example['sentence1']
            new_example['sentence1_lex'] = example['sentence2_lex'] 
            new_example['sentence2_lex'] = example['sentence1_lex'] 
            new_example['gold_label'] = 'neutral'
            new_example['depth'] = example['depth']
            if new_example['sentence1']+new_example['sentence2'] not in example_dedup:
                pmonli_neutral.append(new_example)
                example_dedup.add(new_example['sentence1']+new_example['sentence2'])

        for example in pmonli_neutral:
            if example['sentence1']+example['sentence2'] in example_dedup:
                continue
            example_dedup.add(example['sentence1']+example['sentence2'])
            new_example = {}
            new_example['sentence1'] = example['sentence2']
            new_example['sentence2'] = example['sentence1']
            new_example['sentence1_lex'] = example['sentence2_lex'] 
            new_example['sentence2_lex'] = example['sentence1_lex'] 
            new_example['gold_label'] = 'entailment'
            new_example['depth'] = example['depth']
            if new_example['sentence1']+new_example['sentence2'] not in example_dedup:
                pmonli_entail.append(new_example)
                example_dedup.add(new_example['sentence1']+new_example['sentence2'])

        for example in nmonli_entail:
            if example['sentence1']+example['sentence2'] in example_dedup:
                continue
            example_dedup.add(example['sentence1']+example['sentence2'])
            new_example = {}
            new_example['sentence1'] = example['sentence2']
            new_example['sentence2'] = example['sentence1']
            new_example['sentence1_lex'] = example['sentence2_lex'] 
            new_example['sentence2_lex'] = example['sentence1_lex'] 
            new_example['gold_label'] = 'neutral'
            new_example['depth'] = example['depth']
            if new_example['sentence1']+new_example['sentence2'] not in example_dedup:
                nmonli_neutral.append(new_example)
                example_dedup.add(new_example['sentence1']+new_example['sentence2'])

        for example in nmonli_neutral:
            if example['sentence1']+example['sentence2'] in example_dedup:
                continue
            example_dedup.add(example['sentence1']+example['sentence2'])
            new_example = {}
            new_example['sentence1'] = example['sentence2']
            new_example['sentence2'] = example['sentence1']
            new_example['sentence1_lex'] = example['sentence2_lex'] 
            new_example['sentence2_lex'] = example['sentence1_lex'] 
            new_example['gold_label'] = 'entailment'
            new_example['depth'] = example['depth']
            if new_example['sentence1']+new_example['sentence2'] not in example_dedup:
                nmonli_entail.append(new_example)
                example_dedup.add(new_example['sentence1']+new_example['sentence2'])

        word_entail = pmonli_entail + nmonli_neutral
        word_neutral = pmonli_neutral + nmonli_entail
        word_pos = pmonli_entail + pmonli_neutral # 1
        word_neg = nmonli_neutral + nmonli_entail # 0
        all_examples = word_entail + word_neutral

        word_to_sentences_mapping = {}
        for example in all_examples:
            if example["sentence1_lex"] in word_to_sentences_mapping:
                word_to_sentences_mapping[example["sentence1_lex"]].add(example['sentence1'])
            else:
                word_to_sentences_mapping[example["sentence1_lex"]] = set([example['sentence1']])
            if example["sentence2_lex"] in word_to_sentences_mapping:
                word_to_sentences_mapping[example["sentence2_lex"]].add(example['sentence2'])
            else:
                word_to_sentences_mapping[example["sentence2_lex"]] = set([example['sentence2']])

        entail_pairs = list(set((pair["sentence1_lex"], pair["sentence2_lex"]) for pair in word_entail))
        neutral_pairs = list(set((pair["sentence1_lex"], pair["sentence2_lex"]) for pair in word_neutral))

        word_to_entail_mapping = {}
        for pair in entail_pairs:
            if pair[0] in word_to_entail_mapping:
                word_to_entail_mapping[pair[0]].append(pair[1])
            else:
                word_to_entail_mapping[pair[0]] = [pair[1]]

        word_to_neutral_mapping = {}
        for pair in neutral_pairs:
            if pair[0] in word_to_neutral_mapping:
                word_to_neutral_mapping[pair[0]].append(pair[1])
            else:
                word_to_neutral_mapping[pair[0]] = [pair[1]]
        all_vocab = list(word_to_sentences_mapping.keys())
        
        data = []

        def get_intervention():
            return 1
        
        label_flipped_data = []
        label_unflipped_data = []
        
        while len(label_flipped_data) < self.size:
            pos = random.choice([True, False])
            if pos:
                example = random.choice(word_pos)
                entail = random.choice([True, False])
                if entail and example['sentence1_lex'] in word_to_entail_mapping:
                    entail_candidates = word_to_entail_mapping[example['sentence1_lex']]
                    entail_candidate = random.choice(entail_candidates)
                    # make it super hard, an random template?
                    random_template_word = random.choice(all_vocab)
                    sentence2 = random.choice(list(word_to_sentences_mapping[random_template_word]))
                    has_dot = True if "." in sentence2 else False
                    sentence2_update = sentence2.strip(".").split()
                    for i in range(len(sentence2_update)):
                        if sentence2_update[i] == random_template_word:
                            sentence2_update[i] = entail_candidate
                    if has_dot:
                        sentence2_update[-1] += "."
                    sentence2 = " ".join(sentence2_update)
                    sentence1 = sentence2.strip(".").split()
                    for i in range(len(sentence1)):
                        if sentence1[i] == entail_candidate:
                            residual_vocab = list(set(all_vocab) - set([entail_candidate]))
                            random_base = random.choice(residual_vocab)
                            sentence1[i] = random_base
                    if has_dot:
                        sentence1[-1] += "."
                    sentence1 = " ".join(sentence1)
                    example2 = {}
                    example2['sentence1'] = sentence1
                    example2['sentence2'] = sentence2
                    base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
                    source, source_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
                    base_label = self.ENTAIL_LABEL if example['gold_label'] == 'entailment' else self.NEUTRAL_LABEL
                    intervention = get_intervention()
                    IIT_label = self.ENTAIL_LABEL
#                     print(example, example2, IIT_label)
#                     print("===")
                    if IIT_label == base_label:
                        label_unflipped_data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))
                    else:
                        label_flipped_data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))
                elif not entail and example['sentence1_lex'] in word_to_neutral_mapping:
                    entail_candidates = word_to_neutral_mapping[example['sentence1_lex']]
                    entail_candidate = random.choice(entail_candidates)
                    # make it super hard, an random template?
                    random_template_word = random.choice(all_vocab)
                    sentence2 = random.choice(list(word_to_sentences_mapping[random_template_word]))
                    has_dot = True if "." in sentence2 else False
                    sentence2_update = sentence2.strip(".").split()
                    for i in range(len(sentence2_update)):
                        if sentence2_update[i] == random_template_word:
                            sentence2_update[i] = entail_candidate
                    if has_dot:
                        sentence2_update[-1] += "."
                    sentence2 = " ".join(sentence2_update)
                    sentence1 = sentence2.strip(".").split()
                    for i in range(len(sentence1)):
                        if sentence1[i] == entail_candidate:
                            residual_vocab = list(set(all_vocab) - set([entail_candidate]))
                            random_base = random.choice(residual_vocab)
                            sentence1[i] = random_base
                    if has_dot:
                        sentence1[-1] += "."
                    sentence1 = " ".join(sentence1)
                    example2 = {}
                    example2['sentence1'] = sentence1
                    example2['sentence2'] = sentence2
                    base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
                    source, source_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
                    base_label = self.ENTAIL_LABEL if example['gold_label'] == 'entailment' else self.NEUTRAL_LABEL
                    intervention = get_intervention()
                    IIT_label = self.NEUTRAL_LABEL
#                     print(example, example2, IIT_label)
#                     print("===")
                    if IIT_label == base_label:
                        label_unflipped_data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))
                    else:
                        label_flipped_data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))
            else:
                example = random.choice(word_neg)
                entail = random.choice([True, False])
                if entail and example['sentence1_lex'] in word_to_entail_mapping:
                    entail_candidates = word_to_entail_mapping[example['sentence1_lex']]
                    entail_candidate = random.choice(entail_candidates)
                    # make it super hard, an random template?
                    random_template_word = random.choice(all_vocab)
                    sentence2 = random.choice(list(word_to_sentences_mapping[random_template_word]))
                    has_dot = True if "." in sentence2 else False
                    sentence2_update = sentence2.strip(".").split()
                    for i in range(len(sentence2_update)):
                        if sentence2_update[i] == random_template_word:
                            sentence2_update[i] = entail_candidate
                    if has_dot:
                        sentence2_update[-1] += "."
                    sentence2 = " ".join(sentence2_update)
                    sentence1 = sentence2.strip(".").split()
                    for i in range(len(sentence1)):
                        if sentence1[i] == entail_candidate:
                            residual_vocab = list(set(all_vocab) - set([entail_candidate]))
                            random_base = random.choice(residual_vocab)
                            sentence1[i] = random_base
                    if has_dot:
                        sentence1[-1] += "."
                    sentence1 = " ".join(sentence1)
                    example2 = {}
                    example2['sentence1'] = sentence1
                    example2['sentence2'] = sentence2
                    base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
                    source, source_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
                    base_label = self.ENTAIL_LABEL if example['gold_label'] == 'entailment' else self.NEUTRAL_LABEL
                    intervention = get_intervention()
                    IIT_label = self.NEUTRAL_LABEL
#                     print(example, example2, IIT_label)
#                     print("===")
                    if IIT_label == base_label:
                        label_unflipped_data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))
                    else:
                        label_flipped_data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))
                elif not entail and example['sentence1_lex'] in word_to_neutral_mapping:
                    entail_candidates = word_to_neutral_mapping[example['sentence1_lex']]
                    entail_candidate = random.choice(entail_candidates)
                    # make it super hard, an random template?
                    random_template_word = random.choice(all_vocab)
                    sentence2 = random.choice(list(word_to_sentences_mapping[random_template_word]))
                    has_dot = True if "." in sentence2 else False
                    sentence2_update = sentence2.strip(".").split()
                    for i in range(len(sentence2_update)):
                        if sentence2_update[i] == random_template_word:
                            sentence2_update[i] = entail_candidate
                    if has_dot:
                        sentence2_update[-1] += "."
                    sentence2 = " ".join(sentence2_update)
                    sentence1 = sentence2.strip(".").split()
                    for i in range(len(sentence1)):
                        if sentence1[i] == entail_candidate:
                            residual_vocab = list(set(all_vocab) - set([entail_candidate]))
                            random_base = random.choice(residual_vocab)
                            sentence1[i] = random_base
                    if has_dot:
                        sentence1[-1] += "."
                    sentence1 = " ".join(sentence1)
                    example2 = {}
                    example2['sentence1'] = sentence1
                    example2['sentence2'] = sentence2
                    base, base_mask = self.embed_func([example["sentence1"], example["sentence2"]])
                    source, source_mask = self.embed_func([example2["sentence1"], example2["sentence2"]])
                    base_label = self.ENTAIL_LABEL if example['gold_label'] == 'entailment' else self.NEUTRAL_LABEL
                    intervention = get_intervention()
                    IIT_label = self.ENTAIL_LABEL
#                     print(example, example2, IIT_label)
#                     print("===")
                    if IIT_label == base_label:
                        label_unflipped_data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))
                    else:
                        label_flipped_data.append((base, base_mask, base_label, source, source_mask, IIT_label, intervention))

        random.shuffle(label_flipped_data)
        random.shuffle(label_unflipped_data)
        data = label_flipped_data[:self.size//2] + label_unflipped_data[:self.size//2]
        random.shuffle(data)
        
        base, base_mask, y, source, source_mask, IIT_y, interventions = zip(*data)
        self.base = base
        self.base_mask = base_mask
        self.source = source
        self.source_mask = source_mask
        self.y = np.array(y)
        self.IIT_y = np.array(IIT_y)
        self.interventions = np.array(interventions)
        return (self.base, self.base_mask), self.y, [(self.source,self.source_mask)], self.IIT_y, self.interventions   
                        
    
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
        random.shuffle(data)
        
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
        random.shuffle(data)
        
        base, base_mask, y, source, source_mask, IIT_y, interventions = zip(*data)
        self.base = base
        self.base_mask = base_mask
        self.source = source
        self.source_mask = source_mask
        self.y = np.array(y)
        self.IIT_y = np.array(IIT_y)
        self.interventions = np.array(interventions)
        return (self.base, self.base_mask), self.y, [(self.source,self.source_mask)], self.IIT_y, self.interventions
    
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
        random.shuffle(data)
        
        base, base_mask, y, source, source_mask, IIT_y, interventions = zip(*data)
        self.base = base
        self.base_mask = base_mask
        self.source = source
        self.source_mask = source_mask
        self.y = np.array(y)
        self.IIT_y = np.array(IIT_y)
        self.interventions = np.array(interventions)
        return (self.base, self.base_mask), self.y, [(self.source,self.source_mask)], self.IIT_y, self.interventions
