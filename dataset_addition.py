from causal_model import CausalModel

def listToInput(l):
    return {"X1":l[0], "X2":l[1], "X3":l[2], "X4":l[3], "X5":l[4],
                "Y1":l[5], "Y2":l[6], "Y3":l[7], "Y4":l[8], "Y5":l[9]}

def inputToList(i):
    variables = ["X5", "X4", "X3", "X2", "X1",
                "Y5", "Y4", "Y3", "Y2", "Y1"]
    return [i[var] for var in variables]

def highlevel_addition_model():
    variables = ["X5", "X4", "X3", "X2", "X1",
                "Y5", "Y4", "Y3", "Y2", "Y1",
                "C4","C3", "C2", "C1",
        "O6", "O5", "O4", "O3", "O2", "O1"]
    values = {variable:[_ for _ in range(10)] for variable in variables}
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
    return CausalModel(variables, values, parents, functions)

def print_pos():
    pos = dict()
    for i in range(1,6):
        pos["X"+str(i)] = (6-i,0)
        pos["Y"+str(i)] = (5.5-i,1)
        pos["O"+str(i)] = (6-i,7)
    pos["O"+str(6)] = (6-6,7)
    pos["C1"] = (6-1,3)
    pos["C2"] = (6-2,4)
    pos["C3"] = (6-3,5)
    pos["C4"] = (6-4,6)
    return pos



def test_highlevel():
    model = highlevel_addition_model()
    pos = print_pos()
    model.print_structure(pos=pos)
    print(model.run_forward())
    input = model.sample_input(mandatory={"C1":1, "C4":1, "O6":1})
    print(model.run_forward(intervention=input))
    model.print_setting(model.run_forward(intervention=input), pos = pos)

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
    return CausalModel(variables, values, parents, functions)

test_highlevel()
