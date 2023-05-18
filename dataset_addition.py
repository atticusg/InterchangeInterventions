from causal_model import CausalModel
import torch
import random

# A path is live if each variable on the path is an actual cause of the variables
# later on the path.

def listToInput(l):
    return {"X1":l[0], "X2":l[1], "X3":l[2], "X4":l[3], "X5":l[4],
                "Y1":l[5], "Y2":l[6], "Y3":l[7], "Y4":l[8], "Y5":l[9]}

def pairToInput(x,y):
    x = str(x)
    y = str(y)
    while len(x) != 5:
        x = "0" + x
    while len(y) != 5:
        y = "0" + y
    return {"X1":float(x[4]), "X2":float(x[3]), "X3":float(x[2]), "X4":float(x[1]), "X5":float(x[0]),
                "Y1":float(y[4]), "Y2":float(y[3]), "Y3":float(y[2]), "Y4":float(y[1]), "Y5":float(y[0])}

def inputToTensor(i):
    variables = ["X5", "X4", "X3", "X2", "X1",
                "Y5", "Y4", "Y3", "Y2", "Y1"]
    return torch.tensor([i[var] for var in variables])

def outputToTensor(o):
    vars = ["O6", "O5", "O4", "O3", "O2", "O1"]
    return torch.tensor([o[var] for var in vars])

def highlevel_tall_model():
    variables = ["X5", "X4", "X3", "X2", "X1",
                "Y5", "Y4", "Y3", "Y2", "Y1",
                "C4","C3", "C2", "C1",
        "O6", "O5", "O4", "O3", "O2", "O1"]
    values = {variable:[_ for _ in range(10)] for variable in variables}
    for i in range(4):
        values["C"+str(i+1)] = [0,1]
    parents = { _ : [] for _ in variables[:10]}

    parents["C1"] = ["X1", "Y1"]
    for i in range(2, 5):
        parents["C"+str(i)] = ["X"+str(i), "Y"+str(i), "C"+str(i-1)]

    parents["O1"] = ["X1", "Y1"]
    for i in range(2, 6):
        parents["O"+str(i)] = ["X"+str(i), "Y"+str(i), "C"+str(i-1)]
    parents["O6"] = ["X5", "Y5","C4"]

    def FILLER():
        return 0

    def C1(x1,y1):
        return int(x1 + y1 > 9)

    def COther(x,y,c):
        return int(x + y + c > 9)

    def O1(x1,y1):
        return (x1 + y1) % 10

    def OOther(x,y,c):
        return (x + y + c) % 10

    def O6(x5,y5,c4):
        return int(x5+y5+c4> 9)

    functions = { _ : FILLER for _ in variables[:10]}

    functions["C1"] = C1
    for i in range(2, 5):
        functions["C"+str(i)] = COther

    functions["O1"] = O1
    for i in range(2, 6):
        functions["O"+str(i)] = OOther

    functions["O6"] = O6
    pos = dict()
    for i in range(1,6):
        pos["X"+str(i)] = (6-i,0)
        pos["Y"+str(i)] = (5.5-i,1)
        pos["O"+str(i)] = (6-i,7)
    pos["O"+str(6)] = (6-6,7)
    pos["C1"] = (6-1.5,3)
    pos["C2"] = (6-2.5,4)
    pos["C3"] = (6-3.5,5)
    pos["C4"] = (6-4.5,6)
    return CausalModel(variables, values, parents, functions,pos=pos)

def highlevel_flat_model():
    variables = ["X5", "X4", "X3", "X2", "X1",
                "Y5", "Y4", "Y3", "Y2", "Y1",
                "C4","C3", "C2", "C1",
        "O6", "O5", "O4", "O3", "O2", "O1"]
    values = {variable:[_ for _ in range(10)] for variable in variables}
    for i in range(4):
        values["C"+str(i+1)] = [0,1]
    parents = { _ : [] for _ in variables[:10]}

    parents["C1"] = ["X1", "Y1"]
    for i in range(2, 5):
        parents["C"+str(i)] = ["X"+str(i), "Y"+str(i), "C"+str(i-1)]

    parents["O1"] = ["X1", "Y1"]
    for i in range(2, 6):
        parents["O"+str(i)] = ["X"+str(i), "Y"+str(i), "C"+str(i-1)]
    parents["O6"] = ["X5", "Y5","C4"]

    def FILLER():
        return 0

    def COther(x,y):
        return int(x + y > 9)

    def O1(x1,y1):
        return (x1 + y1) % 10

    def O2(x2,y2,c1):
        return (x2+y2+c1) % 10

    def O6(x5,y5,c4):
        return int(x5+y5+c4> 9)

    functions = { _ : FILLER for _ in variables[:10]}

    functions["C1"] = C1
    for i in range(2, 5):
        functions["C"+str(i)] = COther

    functions["O1"] = O1
    for i in range(2, 6):
        functions["O"+str(i)] = OOther

    functions["O6"] = O6
    pos = dict()
    for i in range(1,6):
        pos["X"+str(i)] = (6-i,0)
        pos["Y"+str(i)] = (5.5-i,1)
        pos["O"+str(i)] = (6-i,7)
    pos["O"+str(6)] = (6-6,7)
    pos["C1"] = (6-0,3)
    pos["C2"] = (6-1,4)
    pos["C3"] = (6-2,5)
    pos["C4"] = (6-3,6)
    return CausalModel(variables, values, parents, functions,pos=pos)


def highlevel_nocarry_model():
    variables = ["X5", "X4", "X3", "X2", "X1",
                "Y5", "Y4", "Y3", "Y2", "Y1",
        "O6", "O5", "O4", "O3", "O2", "O1"]
    values = {variable:[_ for _ in range(10)] for variable in variables}
    parents = { _ : [] for _ in variables[:10]}

    parents["O1"] = ["X1", "Y1"]
    for i in range(2, 6):
        parents["O"+str(i)] = ["X"+str(i), "Y"+str(i)]
    parents["O6"] = ["X5", "Y5"]

    def FILLER():
        return 0

    def OOther(x,y):
        return (x + y) % 10

    def O6(x5,y5):
        return int(x5+y5> 9)

    functions = { _ : FILLER for _ in variables[:10]}
    for i in range(1, 6):
        functions["O"+str(i)] = OOther
    functions["O6"] = O6
    pos = dict()
    for i in range(1,6):
        pos["X"+str(i)] = (6-i,0)
        pos["Y"+str(i)] = (5.5-i,1)
        pos["O"+str(i)] = (6-i,7)
    pos["O"+str(6)] = (6-6,7)
    return CausalModel(variables, values, parents, functions, pos=pos)

def get_filter(partial_setting):
    def compare(total_setting):
        for var in partial_setting:
            if total_setting[var] != partial_setting[var]:
                return False
        return True
    return compare

def get_path_maxlen_filter(model, lengths):
    def check_path(total_setting):
        input = {var:total_setting[var] for var in model.inputs}
        paths = model.find_live_paths(input)
        m = max([l for l in paths.keys() if len(paths[l]) != 0])
        if m in lengths:
            return True
        return False
    return check_path

def get_specific_path_filter(model, start,end):
    def check_path(total_setting):
        input = {var:total_setting[var] for var in model.inputs}
        paths = model.find_live_paths(input)
        for k in paths:
            for path in paths[k]:
                if path[0]==start and path[-1]==end:
                    return True
        return False
    return check_path

def sample_nine_sum():
    TEMP = [(x,y) for x in range(10) for y in range(10) if x + y == 9]
    return random.sample(TEMP,1)[0]

def sample_nonine_sum():
    TEMP = [(x,y) for x in range(10) for y in range(10) if x + y != 9]
    return random.sample(TEMP,1)[0]

def sample_live_circuit(carry, out):
    def sampler():
        x = 0
        y = 0
        for i in range(1,6):
            xn,yn = None, None
            if i > carry and i < out:
                xn, yn = sample_nine_sum()
            elif i == carry or i == out:
                xn, yn = sample_nonine_sum()
            else:
                xn, yn = random.randint(0,9),random.randint(0,9)
            x += xn * 10**(i-1)
            y += yn * 10**(i-1)
        return pairToInput(x,y)
    return sampler


def get_circuit_samplers():
    result = {"O" + str(i):dict() for i in range(1,7)}
    for c in ["C4", "C3", "C2", "C1"]:
        for o in  ["O6","O5","O4", "O3", "O2", "O1"]:
            if int(c[1])+1 < int(o[1]):
                result[o][c] = sample_live_circuit(int(c[1]),int(o[1]))
    return result

def sampler_to_dataset(sampler, size, model):
    X,y = [],[]
    for _ in range(size):
        input = sampler()
        X.append(inputToTensor(input))
        y.append(outputToTensor(model.run_forward(input)))
    return torch.stack(X), torch.stack(y)


def addition_loss(CE):
    def full_loss(batch_preds, base_labels):
        assert batch_preds.shape[1] == 52
        block_size = 10
        loss = 0
        loss += CE(batch_preds[:, :2], base_labels[:,0])
        for i in range(5):
            loss += CE(batch_preds[:, 2 + block_size*i:2 + block_size*(i+1)], base_labels[:,i+1])
        return loss
    return full_loss

def decode_preds(batch_preds):
    result = {}
    result["O6"] = batch_preds[:, :2].argmax(axis=1)
    for i in range(5):
        result["O"+str(5-i)] = batch_preds[:, 2 + 10*i:12 + 10*i].argmax(axis=1)
    return result



def test_model(model, mandatory=None):
    model.print_structure()
    print("\n\nforwardpass")
    print(model.run_forward())
    input = model.sample_input(mandatory)
    print("\n\nforward pass with input")
    print(model.run_forward(intervention=input))
    model.print_setting(model.run_forward(intervention=input))
    print("\n\npaths")
    print(model.find_live_paths(input))

    input = sample_live_circuit(2,6)
    print("\n\nforward pass with live_circuit_input")
    print(model.run_forward(intervention=input))
    model.print_setting(model.run_forward(intervention=input))
    print("\n\npaths")
    print(model.find_live_paths(input))
