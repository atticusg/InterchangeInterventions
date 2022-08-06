import copy
import torch
import itertools
from sklearn.metrics import classification_report

# TOKENS
nouns = ("pine", "oak", "maple", "birch",
          "rose", "daisy", "tulip", "sunflower",
          "robin", "canary", "sparrow", "penguin",
          "sunfish", "salmon", "flounder", "cod",
          "cat", "dog", "mouse", "goat", "pig")

verbs = ("is a", "has", "is", "can" )


words = nouns + verbs

#ISA classes

plants = ("pine", "oak", "maple", "birch",
          "rose", "daisy", "tulip", "sunflower")

animals = ("robin", "canary", "sparrow", "penguin",
          "sunfish", "salmon", "flounder", "cod",
          "cat", "dog", "mouse", "goat", "pig")

trees = ("pine", "oak", "maple", "birch")

flowers = ("rose", "daisy", "tulip", "sunflower")

birds = ("robin", "canary", "sparrow", "penguin")

fishes = ("sunfish", "salmon", "flounder", "cod")

# is classes

pretty = (*flowers, "canary", "cat")

big = ("oak", "maple", "birch", "sunflower", "salmon", "dog", "goat", "pig")

living = copy.copy(nouns)

green = ("pine")

red = ("maple", "rose")

yellow = ("daisy", "sunflower", "canary", "sunfish")

white = ("birch", "tulip", "penguin", "cod")

twirly = ("birch")

# can classes

grow = copy.copy(nouns)

move = copy.copy(animals)

swim = (*fishes, "penguin")

fly = ("robin", "canary", "sparrow")

walk = ("penguin", "cat", "dog", "mouse", "goat", "pig")

sing = ("canary")

# has classes

leaves = ("oak", "maple", "birch",
          "rose", "daisy", "tulip", "sunflower")

roots = copy.copy(plants)

skin = copy.copy(animals)

legs = (*birds, "dog", "cat", "pig", "goat", "mouse")

bark = copy.copy(trees)

branches = copy.copy(trees)

petals = copy.copy(flowers)

wings = copy.copy(birds)

feathers = copy.copy(birds)

scales = copy.copy(fishes)

gills = copy.copy(fishes)

fur = ("cat", "dog", "mouse", "goat")

# verbs

ISA = [[x] for x in nouns] + [plants, animals, fishes, flowers, trees, birds]

isa = [pretty, big, living, green, red, yellow, white, twirly]

can = [grow, move, swim, fly, walk, sing]

has = [leaves, roots, skin, legs, bark, branches,
        petals, wings, feathers, scales, gills, fur]

classes_tuples = ISA + isa + can + has

classes_strings = [*nouns] + ["plants", "animals",
      "fishes", "flowers", "trees", "birds",
      "pretty", "big", "living", "green", "red", "yellow", "white", "twirly",
      "grow", "move", "swim", "fly", "walk", "sing",
      "leaves", "roots", "skin", "legs", "bark", "branches",
        "petals", "wings", "feathers", "scales", "gills", "fur"
      ]



def classwise_report(y, logits, classes="all"):
    preds = torch.clone(logits)
    preds[preds>0] = 1
    preds[preds<0] = 0
    report = ""
    if classes == "all":
        for clas in classes_strings:
            report += f"CLASSIFICATION FOR CLASS:{clas}"
            index = class_to_index(clas)
            report += "\n"
            report += classification_report(preds[:, index], y[:, index])
            report += "\n\n"
    else:
        for clas in classes:
            print(f"CLASSIFICATION FOR CLASS:{clas}")
            index = class_to_index(clas)
            report += "\n"
            report += classification_report(preds[:, index], y[:, index])
            report += "\n\n"
    return report




def class_to_tuple(clas):
    return classes_tuples[classes_strings.index(clas)]

def class_to_index(clas):
    return classes_strings.index(clas)

def index_to_class(index):
    return classes_strings[index]

def word_to_index(word):
    return words.index(word)

def index_to_word(index):
    return words[index]

def generate_data():
    inputs = itertools.product(nouns, verbs)
    X = [(word_to_index(x[0]), word_to_index(x[1])) for x in inputs]
    y = []
    inputs = itertools.product(nouns, verbs)
    for x in inputs:
        label = []
        for clas in ISA:
            class_label = int(x[0] in clas and x[1] == "is a")
            label.append(class_label)
        for clas in isa:
            class_label = int(x[0] in clas and x[1] == "is")
            label.append(class_label)
        for clas in can:
            class_label = int(x[0] in clas and x[1] == "can")
            label.append(class_label)
        for clas in has:
            class_label = int(x[0] in clas and x[1] == "has")
            label.append(class_label)
        y.append(label)
    return torch.Tensor(X), torch.Tensor(y)

def generate_data_plants_have_roots_and_animals_can_move_and_have_skin():
    X_base = []
    X_source = []
    y_base = []
    interventions = []
    y_iit = []
    inputs = list(itertools.product(nouns, verbs))
    inputs2 = list(itertools.product(copy.copy(nouns), copy.copy(verbs)))
    for base in inputs:
        for source in inputs2:
            X_base.append((word_to_index(base[0]), word_to_index(base[1])))
            X_source.append((word_to_index(source[0]), word_to_index(source[1])))

            interventions.append(0)

            label = []
            for clas in ISA:
                class_label = int(base[0] in clas and base[1] == "is a")
                label.append(class_label)
            for clas in isa:
                class_label = int(base[0] in clas and base[1] == "is")
                label.append(class_label)
            for clas in can:
                class_label = int(base[0] in clas and base[1] == "can")
                label.append(class_label)
            for clas in has:
                class_label = int(base[0] in clas and base[1] == "has")
                label.append(class_label)
            y_base.append(label)

            #IIT labels

            label = []
            blackout_classes = []
            index = 0
            for clas in ISA:
                if index in [class_to_index("animals"), class_to_index("plants")]:
                    class_label = int(source[0] in clas and source[1] == "is a")
                    blackout_classes.append(index)
                else:
                    class_label = int(base[0] in clas and base[1] == "is a")
                label.append(class_label)
                index +=1
            for clas in isa:
                class_label = int(base[0] in clas and base[1] == "is")
                label.append(class_label)
                index +=1
            for clas in can:
                if index == class_to_index("move"):
                    class_label = int(source[0] in clas and source[1] == "can")
                    blackout_classes.append(index)
                else:
                    class_label = int(base[0] in clas and base[1] == "can")
                label.append(class_label)
                index +=1
            for clas in has:
                if index in [class_to_index("skin"), class_to_index("roots")]:
                    class_label = int(source[0] in clas and source[1] == "has")
                    blackout_classes.append(index)
                else:
                    class_label = int(base[0] in clas and base[1] == "has")
                label.append(class_label)
                index +=1
            y_iit.append(label)
    return torch.Tensor(X_base), torch.Tensor(y_base), [torch.Tensor(X_source)],  torch.Tensor(y_iit), torch.Tensor(interventions), blackout_classes
