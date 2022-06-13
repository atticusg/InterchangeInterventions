import numpy as np
import copy
import random
import torch
from utils import randvec

__author__ = "Atticus Geiger"
__version__ = "CS224u, Stanford, Spring 2022"

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

def rand_token_id(token_ids):
    return random.choice(token_ids)


def get_IIT_equality_dataset_both(embed_dim, size, token_ids =None):
    train_dataset = IIT_PremackDatasetBoth(
        embed_dim=embed_dim,
        size=size,
        token_ids=token_ids)
    X_base_train, y_base_train, X_sources_train,  y_IIT_train, interventions = train_dataset.create()
    if token_ids is None:
        X_base_train = torch.tensor(X_base_train)
        X_sources_train = [torch.tensor(X_source_train) for X_source_train in X_sources_train]
    else:
        X_base_train = totuple(X_base_train)
        X_sources_train = [totuple(X_source_train) for X_source_train in X_sources_train]
    y_base_train = torch.tensor(y_base_train)
    y_IIT_train = torch.tensor(y_IIT_train)
    interventions = torch.tensor(interventions)
    return X_base_train, y_base_train, X_sources_train,  y_IIT_train, interventions

def get_IIT_equality_dataset_control13(embed_dim, size, token_ids =None):
    class_size = size/2
    train_dataset = IIT_PremackDatasetControl13(
        embed_dim=embed_dim,
        n_pos=class_size,
        n_neg=class_size,
        token_ids=token_ids)
    X_base_train, y_base_train, X_sources_train,  y_IIT_train, interventions = train_dataset.create()
    if token_ids is None:
        X_base_train = torch.tensor(X_base_train)
        X_sources_train = [torch.tensor(X_source_train) for X_source_train in X_sources_train]
    else:
        X_base_train = totuple(X_base_train)
        X_sources_train = [totuple(X_source_train) for X_source_train in X_sources_train]
    y_base_train = torch.tensor(y_base_train)
    y_IIT_train = torch.tensor(y_IIT_train)
    interventions = torch.tensor(interventions)
    return X_base_train, y_base_train, X_sources_train,  y_IIT_train, interventions


def get_IIT_equality_dataset(variable, embed_dim, size, token_ids =None):
    class_size = size/2
    train_dataset = IIT_PremackDataset(
        variable,
        embed_dim=embed_dim,
        n_pos=class_size,
        n_neg=class_size,
        token_ids=token_ids)
    X_base_train, y_base_train, X_sources_train,  y_IIT_train, interventions = train_dataset.create()
    if token_ids is None:
        X_base_train = torch.tensor(X_base_train)
        X_sources_train = [torch.tensor(X_source_train) for X_source_train in X_sources_train]
    else:
        X_base_train = totuple(X_base_train)
        X_sources_train = [totuple(X_source_train) for X_source_train in X_sources_train]
    y_base_train = torch.tensor(y_base_train)
    y_IIT_train = torch.tensor(y_IIT_train)
    interventions = torch.tensor(interventions)
    return X_base_train, y_base_train,X_sources_train,  y_IIT_train, interventions

class IIT_PremackDataset:

    V1 = 0
    V2 = 1
    POS_LABEL = 1
    NEG_LABEL = 0

    def __init__(self,
                variable,
                embed_dim=50,
                n_pos=500,
                n_neg=500,
                intermediate=False,
                token_ids = None):

        if token_ids is None:
            self.bert = False
        else:
            self.bert = True
            self.token_ids = token_ids
        self.variable = variable
        self.embed_dim = embed_dim
        self.n_pos = n_pos
        self.n_neg = n_neg

        for n, v in ((n_pos, 'n_pos'), (n_neg, 'n_neg')):
            if n % 2 != 0:
                raise ValueError(
                    f"The value of {v} must be even to ensure a balanced "
                    f"split across its two sub-types of the {v} class.")

        self.n_same_same_to_same = int(n_pos / 4)
        self.n_diff_diff_to_same = int(n_neg / 4)
        self.n_same_diff_to_same = int(n_neg / 4)
        self.n_diff_same_to_same = int(n_neg / 4)

        self.n_same_same_to_diff = int(n_neg / 4)
        self.n_diff_diff_to_diff = int(n_neg / 4)
        self.n_same_diff_to_diff = int(n_neg / 4)
        self.n_diff_same_to_diff = int(n_neg / 4)

        self.intermediate = intermediate

    def create(self):
        self.data = []
        self.data += self._create_same_same_to_same()
        self.data += self._create_diff_diff_to_same()
        self.data += self._create_same_diff_to_same()
        self.data += self._create_diff_same_to_same()
        self.data += self._create_same_same_to_diff()
        self.data += self._create_diff_diff_to_diff()
        self.data += self._create_same_diff_to_diff()
        self.data += self._create_diff_same_to_diff()
        random.shuffle(self.data)
        data = self.data.copy()
        if self.bert:
            data = [
                (
                    np.array(x1 + x2),
                    np.array(x3 + x4),
                    np.array(x5 + x6),
                    base_label, IIT_label, intervention
                )
                for (x1, x2,x3,x4,x5,x6), base_label, IIT_label, intervention in data
            ]
        else:
            data = [
                (
                    np.concatenate(x1 + x2),
                    np.concatenate(x3 + x4),
                    np.concatenate(x5 + x6),
                    base_label, IIT_label, intervention
                )
                for (x1, x2,x3,x4,x5,x6), base_label, IIT_label, intervention in data
            ]
        base, source, y, IIT_y, interventions = zip(*data)
        self.base = np.array(base)
        self.source = np.array(source)
        self.y = np.array(y)
        self.IIT_y = np.array(IIT_y)
        self.interventions = np.array(interventions)
        self.sources = list()
        self.sources.append(self.source)
        return self.base, self.y, self.sources, self.IIT_y, self.interventions



    def _create_same_same_to_same(self):
        data = []
        for _ in range(self.n_same_same_to_same):
            base_left = self._create_same_pair()
            base_right = self._create_same_pair()
            base_label = self.POS_LABEL
            if self.variable == "V1":
                source_left = self._create_same_pair()
                source_right = self._create_random_pair()
                IIT_label = self.POS_LABEL
                intervention = self.V1
            if self.variable == "V2":
                source_left = self._create_random_pair()
                source_right = self._create_same_pair()
                intervention = self.V2
                IIT_label = self.POS_LABEL
            rep = (base_left, base_right, source_left,source_right)
            data.append((rep, base_label, IIT_label, intervention))
        return data

    def _create_diff_diff_to_same(self):
        data = []
        for _ in range(self.n_diff_diff_to_same):
            base_left = self._create_diff_pair()
            base_right = self._create_diff_pair()
            base_label = self.POS_LABEL
            if self.variable == "V1":
                source_left = self._create_same_pair()
                source_right = self._create_random_pair()
                IIT_label = self.NEG_LABEL
                intervention = self.V1
            if self.variable == "V2":
                source_left = self._create_random_pair()
                source_right = self._create_same_pair()
                IIT_label = self.NEG_LABEL
                intervention = self.V2
            rep = (base_left, base_right, source_left,source_right)
            data.append((rep, base_label, IIT_label, intervention))
        return data

    def _create_same_diff_to_same(self):
        data = []
        for _ in range(self.n_same_diff_to_same):
            base_left = self._create_same_pair()
            base_right = self._create_diff_pair()
            base_label = self.NEG_LABEL
            if self.variable == "V1":
                source_left = self._create_same_pair()
                source_right = self._create_random_pair()
                IIT_label = self.NEG_LABEL
                intervention = self.V1
            if self.variable == "V2":
                source_left = self._create_random_pair()
                source_right = self._create_same_pair()
                IIT_label = self.POS_LABEL
                intervention = self.V2
            rep = (base_left, base_right, source_left,source_right)
            data.append((rep, base_label, IIT_label, intervention))
        return data

    def _create_diff_same_to_same(self):
        data = []
        for _ in range(self.n_diff_same_to_same):
            base_left = self._create_diff_pair()
            base_right = self._create_same_pair()
            base_label = self.NEG_LABEL
            if self.variable == "V1":
                source_left = self._create_same_pair()
                source_right = self._create_random_pair()
                IIT_label = self.POS_LABEL
                intervention = self.V1
            if self.variable == "V2":
                source_left = self._create_random_pair()
                source_right = self._create_same_pair()
                IIT_label = self.NEG_LABEL
                intervention = self.V2
            rep = (base_left, base_right, source_left,source_right)
            data.append((rep, base_label, IIT_label, intervention))
        return data

    def _create_same_same_to_diff(self):
        data = []
        for _ in range(self.n_same_same_to_diff):
            base_left = self._create_same_pair()
            base_right = self._create_same_pair()
            base_label = self.POS_LABEL
            if self.variable == "V1":
                source_left = self._create_diff_pair()
                source_right = self._create_random_pair()
                IIT_label = self.NEG_LABEL
                intervention = self.V1
            if self.variable == "V2":
                source_left = self._create_random_pair()
                source_right = self._create_diff_pair()
                IIT_label = self.NEG_LABEL
                intervention = self.V2
            rep = (base_left, base_right, source_left,source_right)
            data.append((rep, base_label, IIT_label, intervention))
        return data

    def _create_diff_diff_to_diff(self):
        data = []
        for _ in range(self.n_diff_diff_to_diff):
            base_left = self._create_diff_pair()
            base_right = self._create_diff_pair()
            base_label = self.POS_LABEL
            if self.variable == "V1":
                source_left = self._create_diff_pair()
                source_right = self._create_random_pair()
                IIT_label = self.POS_LABEL
                intervention = self.V1
            if self.variable == "V2":
                source_left = self._create_random_pair()
                source_right = self._create_diff_pair()
                IIT_label = self.POS_LABEL
                intervention = self.V2
            rep = (base_left, base_right, source_left,source_right)
            data.append((rep, base_label, IIT_label, intervention))
        return data

    def _create_same_diff_to_diff(self):
        data = []
        for _ in range(self.n_same_diff_to_diff):
            base_left = self._create_same_pair()
            base_right = self._create_diff_pair()
            base_label = self.NEG_LABEL
            if self.variable == "V1":
                source_left = self._create_diff_pair()
                source_right = self._create_random_pair()
                IIT_label = self.POS_LABEL
                intervention = self.V1
            if self.variable == "V2":
                source_left = self._create_random_pair()
                source_right = self._create_diff_pair()
                IIT_label = self.NEG_LABEL
                intervention = self.V2
            rep = (base_left, base_right, source_left,source_right)
            data.append((rep, base_label, IIT_label, intervention))
        return data

    def _create_diff_same_to_diff(self):
        data = []
        for _ in range(self.n_diff_same_to_diff):
            base_left = self._create_diff_pair()
            base_right = self._create_same_pair()
            base_label = self.NEG_LABEL
            if self.variable == "V1":
                source_left = self._create_diff_pair()
                source_right = self._create_random_pair()
                IIT_label = self.NEG_LABEL
                intervention = self.V1
            if self.variable == "V2":
                source_left = self._create_random_pair()
                source_right = self._create_diff_pair()
                IIT_label = self.POS_LABEL
                intervention = self.V2
            rep = (base_left, base_right, source_left,source_right)
            data.append((rep, base_label, IIT_label, intervention))
        return data

    def _create_random_pair(self):
        if random.choice([True,False]):
            return self._create_same_pair()
        else:
            return self._create_diff_pair()

    def _create_same_pair(self):
        if self.bert:
            vec = rand_token_id(self.token_ids)
        else:
            vec = randvec(self.embed_dim)
        return (vec, vec)

    def _create_diff_pair(self):
        if self.bert:
            while True:
                vec1 = rand_token_id(self.token_ids)
                vec2 = rand_token_id(self.token_ids)
                if not vec1 == vec2:
                    break
        else:
            vec1 = randvec(self.embed_dim)
            vec2 = randvec(self.embed_dim)
            assert not np.array_equal(vec1, vec2)
        return (vec1, vec2)

class IIT_PremackDatasetControl13:
    V1 = 0
    V2 = 1
    control13 = 3
    POS_LABEL = 1
    NEG_LABEL = 0

    def __init__(self,
                embed_dim=50,
                n_pos=500,
                n_neg=500,
                intermediate=False,
                token_ids = None):

        if token_ids is None:
            self.bert = False
        else:
            self.bert = True
            self.token_ids = token_ids

        self.embed_dim = embed_dim
        self.n_pos = n_pos
        self.n_neg = n_neg

        for n, v in ((n_pos, 'n_pos'), (n_neg, 'n_neg')):
            if n % 2 != 0:
                raise ValueError(
                    f"The value of {v} must be even to ensure a balanced "
                    f"split across its two sub-types of the {v} class.")

        self.size = n_pos + n_neg

        self.intermediate = intermediate

    def create(self):
        data = self._create_control13(self.size)
        random.shuffle(data)
        data = data.copy()
        if self.bert:
            data = [
                (
                    np.array(x1 + x2),
                    np.array(x3 + x4),
                    np.array(x5 + x6),
                    base_label, IIT_label, intervention
                )
                for (x1, x2,x3,x4,x5,x6), base_label, IIT_label, intervention in data
            ]
        else:
            data = [
                (
                    np.concatenate(x1 + x2),
                    np.concatenate(x3 + x4),
                    np.concatenate(x5 + x6),
                    base_label, IIT_label, intervention
                )
                for (x1, x2,x3,x4,x5,x6), base_label, IIT_label, intervention in data
            ]
        base, source, y, IIT_y, interventions = zip(*data)
        self.base = np.array(base)
        self.source = np.array(source)
        self.y = np.array(y)
        self.IIT_y = np.array(IIT_y)
        self.interventions = np.array(interventions)
        self.sources = list()
        self.sources.append(self.source)
        return self.base, self.y, self.sources, self.IIT_y, self.interventions

    def _create_control13(self, size):
        data = []
        for _ in range(int(size)):
            base_left = self._create_random_pair()
            base_right = self._create_random_pair()
            intervention = self.control13
            if (base_left[0] == base_left[1]).all() == (base_right[0] == base_right[1]).all():
                base_label = self.POS_LABEL
            else:
                base_label = self.NEG_LABEL

            if random.choice([True,False]):
                source_left = self._create_random_pair()
                source_left = (copy.deepcopy(base_left[1]), source_left[1])
            else:
                source_left = self._create_random_pair()

            if random.choice([True,False]):
                source_right = self._create_random_pair()
                source_right = (copy.deepcopy(base_right[1]), source_right[1])
            else:
                source_right = self._create_random_pair()

            rep = (base_left, base_right, source_left, source_right)

            if (source_left[0] == base_left[1]).all() == (source_right[0] == base_right[1]).all():
                IIT_label = self.POS_LABEL
            else:
                IIT_label = self.NEG_LABEL

            data.append((rep, base_label, IIT_label, intervention))
        return data

    def _create_random_pair(self):
        if random.choice([True,False]):
            return self._create_same_pair()
        else:
            return self._create_diff_pair()

    def _create_same_pair(self):
        if self.bert:
            vec = rand_token_id(self.token_ids)
        else:
            vec = randvec(self.embed_dim)
        return (vec, vec)

    def _create_diff_pair(self):
        if self.bert:
            while True:
                vec1 = rand_token_id(self.token_ids)
                vec2 = rand_token_id(self.token_ids)
                if not vec1 == vec2:
                    break
        else:
            vec1 = randvec(self.embed_dim)
            vec2 = randvec(self.embed_dim)
            assert not np.array_equal(vec1, vec2)
        return (vec1, vec2)

class IIT_PremackDatasetBoth:

    V1 = 0
    V2 = 1
    POS_LABEL = 1
    NEG_LABEL = 0
    both_coord_id = 2

    def __init__(self,
                size= 1000,
                embed_dim=50,
                intermediate=False,
                token_ids = None):

        if token_ids is None:
            self.bert = False
        else:
            self.bert = True
            self.token_ids = token_ids

        self.embed_dim = embed_dim
        self.size= size


        self.intermediate = intermediate

    def create(self):
        data = []
        for _ in range(self.size):
            rep = [self._create_random_pair() for _ in range(6)]
            if self.bert:
                if (rep[0][0] == rep[0][1]) == (rep[1][0] == rep[1][1]):
                    base_label = self.POS_LABEL
                else:
                    base_label = self.NEG_LABEL
                if (rep[2][0] == rep[2][1]) == (rep[5][0] == rep[5][1]):
                    IIT_label = self.POS_LABEL
                else:
                    IIT_label = self.NEG_LABEL

            else:
                if (rep[0][0] == rep[0][1]).all() == (rep[1][0] == rep[1][1]).all():
                    base_label = self.POS_LABEL
                else:
                    base_label = self.NEG_LABEL
                if (rep[2][0] == rep[2][1]).all() == (rep[5][0] == rep[5][1]).all():
                    IIT_label = self.POS_LABEL
                else:
                    IIT_label = self.NEG_LABEL
            data.append((rep,base_label, IIT_label, self.both_coord_id))
        random.shuffle(data)
        data = data.copy()
        if self.bert:
            data = [
                (
                    np.array(x1 + x2),
                    np.array(x3 + x4),
                    np.array(x5 + x6),
                    base_label, IIT_label, intervention
                )
                for (x1, x2,x3,x4,x5,x6), base_label, IIT_label, intervention in data
            ]
        else:
            data = [
                (
                    np.concatenate(x1 + x2),
                    np.concatenate(x3 + x4),
                    np.concatenate(x5 + x6),
                    base_label, IIT_label, intervention
                )
                for (x1, x2,x3,x4,x5,x6), base_label, IIT_label, intervention in data
            ]
        base, source, source2, y, IIT_y, interventions = zip(*data)
        self.base = np.array(base)
        self.source = np.array(source)
        self.source2 = np.array(source2)
        self.y = np.array(y)
        self.IIT_y = np.array(IIT_y)
        self.interventions = np.array(interventions)
        return self.base, self.y,[self.source, self.source2],  self.IIT_y, self.interventions


    def _create_random_pair(self):
        if random.choice([True,False]):
            return self._create_same_pair()
        else:
            return self._create_diff_pair()

    def _create_same_pair(self):
        if self.bert:
            vec = rand_token_id(self.token_ids)
        else:
            vec = randvec(self.embed_dim)
        return (vec, vec)

    def _create_diff_pair(self):
        if self.bert:
            while True:
                vec1 = rand_token_id(self.token_ids)
                vec2 = rand_token_id(self.token_ids)
                if not vec1 == vec2:
                    break
        else:
            vec1 = randvec(self.embed_dim)
            vec2 = randvec(self.embed_dim)
            assert not np.array_equal(vec1, vec2)
        return (vec1, vec2)
