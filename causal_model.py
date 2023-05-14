import random
import copy
import inspect
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

class CausalModel:
    def __init__(self, variables, values, parents, functions, timesteps = None, pos = None):
        self.variables = variables
        self.values= values
        self.parents = parents
        self.children = {var:[] for var in variables}
        for variable in variables:
            assert variable in self.parents
            for parent in self.parents[variable]:
                self.children[parent].append(variable)
        self.functions = functions
        self.start_variables = []
        self.timesteps = timesteps
        for variable in self.variables:
            assert variable in self.values
            assert variable in self.children
            assert variable in self.functions
            assert len(inspect.getfullargspec(self.functions[variable])[0]) == len(self.parents[variable])
            if timesteps is not None:
                assert variable in timesteps
            for variable2 in copy.copy(self.variables):
                if variable2 in self.parents[variable]:
                    assert variable in self.children[variable2]
                    if timesteps is not None:
                        assert timesteps[variable2] < timesteps[variable]
                if variable2 in self.children[variable]:
                    assert variable in parents[variable2]
                    if timesteps is not None:
                        assert timesteps[variable2] > timesteps[variable]
            if len(self.parents) == 0:
                self.start_variables.add(variable)

        self.inputs = [ var  for var in self.variables if len(parents[var])==0]
        self.outputs = copy.deepcopy(variables)
        for child in variables:
            for parent in parents[child]:
                if parent in self.outputs:
                    self.outputs.remove(parent)
        if self.timesteps is not None:
            self.timesteps = timesteps
        else:
            self.timesteps,self.end_time = self.generate_timesteps()
            for output in self.outputs:
                self.timesteps[output] = self.end_time
        self.variables.sort(key=lambda x: self.timesteps[x])
        self.run_forward()
        self.pos = pos
        width = {_:0 for _ in range(len(self.variables))}
        if self.pos == None:
            self.pos = dict()
        for var in self.variables:
            if var not in pos:
                pos[var] = (width[self.timesteps[var]], self.timesteps[var])
                width[self.timesteps[var]] += 1


    def generate_timesteps(self):
        timesteps = {input:0 for input in self.inputs}
        step = 1
        change = True
        while change:
            change = False
            copytimesteps = copy.deepcopy(timesteps)
            for parent in timesteps:
                if timesteps[parent] == step-1:
                    for child in self.children[parent]:
                        copytimesteps[child] = step
                        change = True
            timesteps = copytimesteps
            step += 1
        for var in self.variables:
            assert var in timesteps
        return timesteps, step-1

    def marginalize(self, tumor):
        #TODO
        for var in tumor:
            return None

    def print_structure(self,pos=None):
        G = nx.DiGraph()
        G.add_edges_from([(parent,child) for child in self.variables for parent in self.parents[child]])
        plt.figure(figsize=(10,10))
        edges = nx.draw_networkx(G, with_labels = True, node_color ='green', pos = self.pos)
        plt.show()

    def find_live_paths(self, intervention):
        actual_setting = self.run_forward(intervention)
        paths = [[variable] for variable in self.variables]
        while True:
            new_paths = []
            for path in paths:
                for child in self.children[path[-1]]:
                    actual_cause = False
                    for value in self.values[path[-1]]:
                        newintervention = copy.deepcopy(intervention)
                        newintervention[path[-1]] = value
                        counterfactual_setting = self.run_forward(newintervention)
                        if counterfactual_setting[child] != actual_setting[child]:
                            actual_cause = True
                    if actual_cause:
                        new_paths.append(copy.deepcopy(path)+[child])
            paths = new_paths
            if len(paths) == len(new_paths):
                break
        return [path for path in paths if len(path)>1]


    def print_setting(self,total_setting):
        relabeler = {var: var + ": " + str(total_setting[var]) for var in self.variables}
        G = nx.DiGraph()
        G.add_edges_from([(parent,child) for child in self.variables for parent in self.parents[child]])
        plt.figure(figsize=(10,10))
        G = nx.relabel_nodes(G, relabeler)
        newpos = dict()
        if self.pos is not None:
            for var in self.pos:
                newpos[relabeler[var]] = self.pos[var]
        edges = nx.draw_networkx(G, with_labels = True, node_color ='green', pos = newpos)
        plt.show()


    def run_forward(self, intervention = None):
        total_setting = defaultdict(None)
        length = len(list(total_setting.keys()))
        step = 0
        while length != len(self.variables):
            for variable in self.variables:
                for variable2 in self.parents[variable]:
                    if variable2 not in total_setting:
                        continue
                if intervention is not None and variable in intervention:
                    total_setting[variable] = intervention[variable]
                else:
                    total_setting[variable] = self.functions[variable](*[total_setting[parent] for parent in self.parents[variable]])
            length = len(list(total_setting.keys()))
        return total_setting

    def add_variable(self, variable, values, parents, children, function, timestep=None):
        if timestep is not None:
            assert self.timesteps is not None
            self.timesteps[variable] = timestep
        for parent in parents:
            assert parent in self.variables
        for child in children:
            assert child in self.variables
        self.parents[variable] = parents
        self.children[variable] = children
        self.values[variable] = values
        self.functions[variable] = function

    def sample_input(self, mandatory=None):
        input = {var: random.sample(self.values[var],1)[0] for var in self.inputs}
        total = self.run_forward(intervention=input)
        def agree(t, m):
            for var in m:
                if m[var] != t[var]:
                    return False
            return True
        while mandatory is not None and not agree(total, mandatory):
            input = {var: random.sample(self.values[var],1)[0] for var in self.inputs}
            total = self.run_forward(intervention=input)
        return input

def simple_example():
    variables = ["A", "B", "C"]
    values= {variable:[True, False] for variable in variables}
    parents = {"A":[], "B":[], "C":["A", "B"]}
    def A():
        return True
    def B():
        return False
    def C(a,b):
        return a and b
    functions = {"A": A, "B": B, "C": C}
    model = CausalModel(variables, values, parents, functions)
    model.print_structure()
    print("No intervention:\n", model.run_forward(), "\n")
    model.print_setting(model.run_forward())
    print("Intervention setting A and B to TRUE:\n", model.run_forward({"A":True, "B":True}))
    print("Timesteps:", model.timesteps)



if __name__ == '__main__':
    simple_example()
