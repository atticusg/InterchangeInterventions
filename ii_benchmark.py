import torch
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertModel

from trainer import LIMTrainer, BERTLIMTrainer
from LIM_deep_neural_classifier import LIMDeepNeuralClassifier
from LIM_bert import LIMBERTClassifier
import dataset_equality
import dataset_nli

import utils


class IIBenchmark:
    def __init__(
        self,
        variable_names,
        data_parameters={},
        model_parameters={},
        training_parameters={},
        seed=42
    ):
        self.variable_names = variable_names
        self.data_parameters = data_parameters
        self.model_parameters = model_parameters
        self.training_parameters = training_parameters
        self.seed = seed

        utils.fix_random_seeds()

        self.train_dataset, self.test_dataset = self.load_datasets()


    def load_datasets(self):
        return


    def create_model(self):
        return


    def create_classifier(self, model):
        return


    def evaluate(self, model, alignment):
        """
        Evaluates an alignment for a Layered Intervenable Model (LIM) on the
        interchange-interventions objective.

        Returns
        --------
        tuple of size = (number of variable, 2), with the following format:
            (
                (y_true for V1 alignment, y_pred for V1 alignment),
                (y_true for V2 alignment, y_pred for V2 alignment),
                ...
            )
        """
        result = []
        for i in range(len(self.test_dataset)):
            X_base_test, y_base_test, X_sources_test, y_IIT_test, interventions = self.test_dataset[i]

            LIM_trainer = self.create_classifier(model)

            IIT_preds = LIM_trainer.iit_predict(X_base_test,
                                                X_sources_test,
                                                interventions,
                                                alignment, device='cpu')

            result += [(y_IIT_test, IIT_preds)]

        return tuple(result)


    def display_evaluations(self, evaluations):
        """
        Displays evaluations from `evaluate` function.
        """
        for e, var_name in zip(evaluations, VARIABLE_NAMES):
            print(f'II-Evaluation on {var_name}')
            print(classification_report(*e))


    def train_model(self, variable, alignment):
        """
        Trains an LIM model using IIT on a variable (V1, V2, or BOTH)
        and an alignment.
        Used to train models for benchmark.
        """
        # set up model
        LIM = self.create_model()

        # wrap trainer around model
        LIM_trainer = self.create_classifier(LIM)

        # choose training data based off of intervention variable (V1, V2, or both)
        X_base_train, y_base_train, X_sources_train, y_IIT_train, interventions = self.train_dataset[variable]

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


    def get_alignments_for_layer(self, layer):
        possible_alignments = [
            {'layer': layer, 'start': i, 'end': i + 1}
            for i in range(self.model_parameters['hidden_dim'])
        ]
        return possible_alignments


    def sample_alignments(self, layers=[1, 2], sizes=[2, 4, 6, 8], n_samples=10):
        samples = []
        for _ in range(n_samples):
            # 1. sample layer for all variables
            layer = np.random.choice(layers)
            possible_alignments = self.get_alignments_for_layer(layer)

            # 2. sample span sizes for each variable
            # NOTE: assume that max in sizes is less than or equal to half of hidden dimension
            n_v1 = np.random.choice(sizes)
            n_v2 = np.random.choice(sizes)

            # 3. sample alignments for the two variables
            sample = np.random.choice(a=possible_alignments, size=(n_v1 + n_v2))

            samples.append({
                V1: sample[:n_v1],
                V2: sample[n_v1:],
                BOTH: sample
            })

        return samples


    def train_models(self, alignments, name='equality'):
        """
        Train benchmark models using IIT for each model in the list of alignments.
        For each alignment, trains 3 separate models: aligning only V1, aligning only V2, and aligning BOTH.
        """
        for alignment in alignments:
            for i, variable in enumerate(self.variable_names):
                LIM_model = self.train_model(i, alignment)
                torch.save(
                    LIM_model.state_dict(),
                    f"./models/{name}-{variable}-{i:0>2d}.pt"
                )


    def load_model(self, path):
        """
        Load LIM model from saved path.
        Assumes all model parameters have been kept constants.
        """
        LIM_model = self.create_model()
        weights = torch.load(path)
        LIM_model.load_state_dict(state_dict=weights)
        return LIM_model


class IIBenchmarkEquality(IIBenchmark):
    def __init__(
        self,
        variable_names=['V1', 'V2', 'BOTH'],
        data_parameters={
            'train_size': 100000, 'test_size': 1000, 'embedding_dim': 4
        },
        model_parameters={
            'num_layers': 3, 'hidden_dim': 16, 'hidden_activation': torch.nn.ReLU(), 'input_dim': 16, 'n_classes': 2
        },
        training_parameters={
            'warm_start': True, 'max_iter': 10, 'batch_size': 64, 'n_iter_no_change': 10000,
            'shuffle_train': False, 'eta': 0.001
        },
        seed=42
    ):
        super().__init__(
            variable_names,
            data_parameters,
            model_parameters,
            training_parameters,
            seed
        )

    def load_datasets(self):
        embedding_dim = self.data_parameters['embedding_dim']
        train_size = self.data_parameters['train_size']
        test_size = self.data_parameters['test_size']

        iit_equality_train_v1 = dataset_equality.get_IIT_equality_dataset("V1", embedding_dim, train_size)
        iit_equality_train_v2 = dataset_equality.get_IIT_equality_dataset("V2", embedding_dim, train_size)
        iit_equality_train_both = dataset_equality.get_IIT_equality_dataset_both(embedding_dim, train_size)

        iit_equality_train = [iit_equality_train_v1, iit_equality_train_v2, iit_equality_train_both]

        iit_equality_test_v1 = dataset_equality.get_IIT_equality_dataset("V1", embedding_dim, test_size)
        iit_equality_test_v2 = dataset_equality.get_IIT_equality_dataset("V2", embedding_dim, test_size)
        iit_equality_test_both = dataset_equality.get_IIT_equality_dataset_both(embedding_dim, test_size)

        iit_equality_test = [iit_equality_test_v1, iit_equality_test_v2, iit_equality_test_both]

        return iit_equality_train, iit_equality_test


    def create_model(self):
        return LIMDeepNeuralClassifier(
            **self.model_parameters
        )


    def create_classifier(self, model):
        return LIMTrainer(
            model,
            **self.training_parameters
        )


class IIBenchmarkMoNli(IIBenchmark):
    def __init__(
        self,
        variable_names=['LEX'],
        data_parameters={
            'train_size': 10000, 'test_size': 10000
        },
        model_parameters={
            'weights_name': 'bert-base-uncased', 'max_length': 40, 'n_classes': 2, 'hidden_dim': 768
        },
        training_parameters={
            'warm_start': True, 'max_iter': 5, 'batch_size': 16, 'n_iter_no_change': 10000,
            'shuffle_train': False, 'eta': 0.0001
        },
        seed=42
    ):
        super().__init__(
            variable_names,
            data_parameters,
            model_parameters,
            training_parameters,
            seed
        )

    def load_datasets(self):
        bert_tokenizer = BertTokenizer.from_pretrained(self.model_parameters['weights_name'])

        def encoding(X):
            if X[0][-1] != ".":
                input = [". ".join(X)]
            else:
                input = X
            data = bert_tokenizer.batch_encode_plus(
                    input,
                    max_length=self.model_parameters['max_length'],
                    add_special_tokens=True,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True)
            indices = torch.tensor(data['input_ids'])
            mask = torch.tensor(data['attention_mask'])
            return (indices, mask)

        iit_MoNLI_train = dataset_nli.get_IIT_MoNLI_dataset(encoding, 'train', self.data_parameters['train_size'])
        iit_MoNLI_test = dataset_nli.get_IIT_MoNLI_dataset(encoding, 'test', self.data_parameters['test_size'])
        return iit_MoNLI_train, iit_MoNLI_test


    def create_model(self):
        bert = BertModel.from_pretrained(self.model_parameters['weights_name'])

        return LIMBERTClassifier(
            self.model_parameters['n_classes'],
            bert,
            self.model_parameters['max_length'],
            debug=self.model_parameters['debug'],
            target_dims = self.model_parameters['target_dims'],
            target_layers=self.model_parameters['target_layers'],
        )


    def create_classifier(self, model):
        return BERTLIMTrainer(
            model,
            **self.training_parameters
        )
