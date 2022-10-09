from operator import ge
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import utils
from LIM_deep_neural_classifier import LIMDeepNeuralClassifier
import dataset_equality
from trainer import LIMTrainer

utils.fix_random_seeds()

################ CONSTANTS ####################
SEED = 42
VARIABLE_NAMES = ['V1', 'V2', 'BOTH']
V1 = 0
V2 = 1
BOTH = 2

# data specification
embedding_dim = 4
train_size = 100000
test_size = 1000

# model parameters
num_layers = 3
hidden_dim = embedding_dim * 4
hidden_activation = torch.nn.ReLU()
input_dim = embedding_dim * 4
n_classes = 2

# training hyperparameters
warm_start = True
max_iter = 10
batch_size = 64
n_iter_no_change = 10000
shuffle_train = False
eta = 0.001
###############################################

# load training and testing dataset (training is to generate models)
iit_equality_train_v1 = dataset_equality.get_IIT_equality_dataset("V1", embedding_dim, train_size)
iit_equality_train_v2 = dataset_equality.get_IIT_equality_dataset("V2", embedding_dim, train_size)
iit_equality_train_both = dataset_equality.get_IIT_equality_dataset_both(embedding_dim, train_size)

iit_equality_train = [iit_equality_train_v1, iit_equality_train_v2, iit_equality_train_both]

iit_equality_test_v1 = dataset_equality.get_IIT_equality_dataset("V1", embedding_dim, test_size)
iit_equality_test_v2 = dataset_equality.get_IIT_equality_dataset("V2", embedding_dim, test_size)
iit_equality_test_both = dataset_equality.get_IIT_equality_dataset_both(embedding_dim, test_size)

iit_equality_test = [iit_equality_test_v1, iit_equality_test_v2, iit_equality_test_both]


def evaluate(LIM, alignment, iit_equality_test=iit_equality_test):
    """
    Evaluates an alignment for a Layered Intervenable Model (LIM) on the 
    interchange-interventions objective.

    Returns
    --------
    tuple of 6 items
        (
            y_true for V1 alignment, y_pred for V1 alignment,
            y_true for V2 alignment, y_pred for V2 alignment,
            y_true for BOTH alignment, y_pred for BOTH alignment
        )
    """
    result = []
    for i in range(len(iit_equality_test)):
        X_base_test, y_base_test, X_sources_test, y_IIT_test, interventions = iit_equality_test[i]

        LIM_trainer = LIMTrainer(LIM)
        
        IIT_preds = LIM_trainer.iit_predict(X_base_test,
                                            X_sources_test,
                                            interventions,
                                            alignment, device='cpu')
        
        result += [(y_IIT_test, IIT_preds)]

    return tuple(result)


def display_evaluations(evaluations):
    """
    Displays evaluations from `evaluate` function.
    """
    for e, var_name in zip(evaluations, VARIABLE_NAMES):
        print(f'II-Evaluation on {var_name}')
        print(classification_report(*e))


def train_model(variable, alignment):
    """
    Trains an LIM model using IIT on a variable (V1, V2, or BOTH)
    and an alignment.
    Used to train models for benchmark.
    """
    # set up model
    LIM = LIMDeepNeuralClassifier(
        hidden_dim=hidden_dim, 
        hidden_activation=hidden_activation, 
        num_layers=num_layers,
        input_dim=input_dim,
        n_classes=n_classes
    )

    # wrap trainer around model
    LIM_trainer = LIMTrainer(
        LIM,
        warm_start=warm_start,
        max_iter=max_iter,
        batch_size=batch_size,
        n_iter_no_change=n_iter_no_change,
        shuffle_train=shuffle_train,
        eta=eta
    )

    # choose training data based off of intervention variable (V1, V2, or both)
    X_base_train, y_base_train, X_sources_train, y_IIT_train, interventions = iit_equality_train[variable]

    # train model
    _ = LIM_trainer.fit(
        X_base_train, 
        y_base_train, 
        iit_data=(
            X_sources_train,
            y_IIT_train,
            interventions
        ),
        intervention_ids_to_coords=alignment
    )

    # return model
    return LIM_trainer.model


def overlap_exists(layer_v1, start_v1, end_v1, layer_v2, start_v2, end_v2):
    """
    Check for overlap between alignment for V1 and alignment for V2
    """
    if layer_v1 != layer_v2:
        return False
    return (start_v1 >= start_v2 and start_v1 < end_v2) or (start_v2 >= start_v1 and start_v2 < end_v1)


def generate_all_alignments(num_layers=num_layers, hidden_dim=hidden_dim):
    """
    Generate all possible alignment combinations for V1 and V2.
    Allows for overlap of ranges, but marks it in a table so that these can be filtered out.

    NOTE: currently assumes that a node aligns with a contiguous block of neurons (cannot
    map a node to an activation in layer 1 and a separate activation in layer 2, or to indices 1:3 and 5:7 in layer 1).
    """
    alignments = []
    lengths_v1 = []
    lengths_v2 = []
    overlaps = []
    for layer_v1 in range(1, num_layers):
        for start_v1 in range(hidden_dim):
            for end_v1 in range(start_v1 + 1, hidden_dim):
                v1_alignment = [{'layer': layer_v1, 'start': start_v1, 'end': end_v1}]
                for layer_v2 in range(1, num_layers):
                    for start_v2 in range(hidden_dim):
                        for end_v2 in range(start_v2 + 1, hidden_dim):
                            v2_alignment = [{'layer': layer_v2, 'start': start_v2, 'end': end_v2}]
                            alignments += [{
                                V1: v1_alignment,
                                V2: v2_alignment,
                                BOTH: v1_alignment + v2_alignment
                            }]
                            lengths_v1 += [end_v1 - start_v1]
                            lengths_v2 += [end_v2 - start_v2]
                            overlaps += [overlap_exists(layer_v1, start_v1, end_v1, layer_v2, start_v2, end_v2)]
    
    return pd.DataFrame({
        'alignment': alignments,
        'length_v1': lengths_v1,
        'length_v2': lengths_v2,
        'overlap': overlaps
    })


def get_possible_alignments(num_layers=num_layers, hidden_dim=hidden_dim):
    possible_alignments = []
    # assumes we don't want alignments at layer 0
    for layer in range(1, num_layers):
        possible_alignments += [{'layer': layer, 'start': i, 'end': i + 1} for i in range(hidden_dim)]
    return possible_alignments


def sample_alignment(n_samples=10):
    possible_alignments = get_possible_alignments()
    n_alignments = len(possible_alignments)
    samples = []
    for _ in n_samples:
        # NOTE: assumption that a variable will not align to more than half of a model's neurons
        n_v1 = np.random.randint(1, n_alignments // 2)
        n_v2 = np.random.randint(1, n_alignments // 2)

        sample = np.random.choice(a=possible_alignments, size=(n_v1 + n_v2))

        samples.append({
            V1: sample[:n_v1],
            V2: sample[n_v1:],
            BOTH: sample
        })
    
    return samples


def sample_alignments(all_alignments, n_samples=10):
    """
    Sample relevant alignments for the II-benchmark.
    Currently filtes for no overlap, and stratifies sampling by lengths: 
    for each length of v1 and length of v2, sample one example.
    NOTE: can try to think of other stratified sampling methods
    """
    stratified_df = \
        all_alignments[~all_alignments['overlap']].groupby(
            ['length_v1', 'length_v2'], group_keys=False
        ).apply(
            lambda x: x.sample(random_state=SEED)
        )
    sampled_alignments = stratified_df.sample(n_samples, random_state=SEED).reset_index(drop=True)
    return sampled_alignments


def train_models(alignments):
    """
    Train benchmark models using IIT for each model in the list of alignments.
    For each alignment, trains 3 separate models: aligning only V1, aligning only V2, and aligning BOTH.
    """
    for alignment in alignments['alignment'].values:
        for variable in [V1, V2, BOTH]:
            LIM_model = train_model(variable, alignment)
            v1 = alignment[V1][0]
            v2 = alignment[V2][0]
            torch.save(
                LIM_model.state_dict(), 
                f"./models/v={variable}v1={v1['layer']}-{v1['start']}-{v1['end']}v2={v2['layer']}-{v2['start']}-{v2['end']}.pt"
            )


def load_model(path):
    """
    Load LIM model from saved path.
    Assumes all model parameters have been kept constants.
    """
    LIM_model = LIMDeepNeuralClassifier(
        hidden_dim=hidden_dim, 
        hidden_activation=hidden_activation, 
        num_layers=num_layers,
        input_dim=input_dim,
        n_classes=n_classes
    )
    weights = torch.load(path)
    LIM_model.load_state_dict(state_dict=weights)
    return LIM_model