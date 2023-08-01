import numpy as np
import copy
import random
import torch
import itertools
from utils import randvec

def create_model(number_of_entities, intermediate_vars= ["WX", "YZ"]):
    #insure that the intermediate variables are viable
    ordering = ["X2", "Y2", "Z2", "W2",
                "WY", "WZ", "XY", "XZ","WX", "YZ",
                "WXY", "WYZ", "XYZ", "WXZ"]

    seen = []
    for var in intermediate_vars:
        assert var in ordering
        for char in var:
            assert char == "2" or char not in seen
            seen.append(char)

    intermediate_vars.sort(key=lambda x:ordering(x))


    #define the input space for every kind of variable
    reps = [rand_vec() for _ in range(number_of_entities)]
    variables = ["W","X", "Y", "Z", "O"] + intermediate_vars
    values = {variable:reps for variable in ["W","X", "Y", "Z"]}
    for far in intermediate_vars:
        if var in ["WX", "YZ"]:
            values[var] = [True, False]
        if var in ["WY", "WZ", "XY", "XZ"]:
            values[var] = intertools.product([reps,reps])
        if var in ["WXY", "WXZ"]:
            values[var] = intertools.product([[True,False],reps])
        if var in ["WYZ", "XYZ"]:
            values[var] = intertools.product([reps,[True,False]])
    values["O"] = [True, False]

    parents = {"O":intermediate_vars, "W":[], "X":[], "Y":[], "Z":[]}
    for var in ["W", "X", "Y", "Z"]:
        if var not in seen:
            parents["O"].append(var)
    for var in intermediate_vars:
        parents[var] = []
        for char in var:
            if char != "2":
                parents[var].append(char)

    functions = {"W":None, "X":None, "Y":None, "Z":None}
    functions["O"] = lambda w,x,y,z: (w == x) == (y == z)
    for var in intermediate_vars:
        if var in ["WX", "YZ"]:
            functions[var] = lambda x,y: x == y
            if len(parents["O"]) == 3:
                functions["O"] = lambda x,y,k: (x == y) == k
            if len(parents["O"]) == 2:
                functions["O"] = lambda k1,k2: k1 == k1
        if var in ["WY", "WZ", "XY", "XZ"]:
            functions[var] = lambda x,y: (x,y)
            if len(parents["O"]) == 3:
                functions["O"] = lambda x,y,k: (x == k[0]) == (y == k[1])
            if len(parents["O"]) == 2:
                functions["O"] = lambda k1,k2: (k1[0] == k2[0]) == (k1[1] == k2[1])
        if var in ["WXY", "WXZ"]:
            functions[var] = lambda x,y,z: (x == y,z)
            functions["O"] = lambda x,k: (x == k[1]) == k[0]
        if var in ["WYZ", "XYZ"]:
            functions[var] = lambda x,y,z: (x,y==z)
            functions["O"] = lambda x,k: (x == k[0]) == k[1]
    return CausalModel(variables, values, parents, functions)



def create_random_pair(self):
    if random.choice([True,False]):
        return self._create_same_pair()
    else:
        return self._create_diff_pair()

def create_same_pair(self):
    if self.bert:
        vec = rand_token_id(self.token_ids)
    else:
        vec = randvec(self.embed_dim)
    return (vec, vec)

def create_diff_pair(self):
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
