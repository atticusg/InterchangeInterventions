import copy
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import utils

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2022"

class LIMTrainer:
    def __init__(self,
            LIM,
            batch_size=1028,
            max_iter=1000,
            eta=0.001,
            optimizer_class=torch.optim.Adam,
            l2_strength=0,
            gradient_accumulation_steps=1,
            max_grad_norm=None,
            warm_start=False,
            early_stopping=False,
            validation_fraction=0.1,
            shuffle_train=True,
            n_iter_no_change=10,
            tol=1e-5,
            device=None,
            display_progress=True,
            **optimizer_kwargs):
        """
        Base class for all the PyTorch-based models.

        Parameters
        ----------
        LIM: `LayeredIntervenableModel`
            The model to be trained.

        batch_size: int
            Number of examples per batch. Batching is handled by a
            `torch.utils.data.DataLoader`. Final batches can have fewer
            examples, depending on the total number of examples in the
            dataset.

        max_iter: int
            Maximum number of training iterations. This will interact
            with `early_stopping`, `n_iter_no_change`, and `tol` in the
            sense that this limit will be reached if and only if and
            conditions triggered by those other parameters are not met.

        eta : float
            Learning rate for the optimizer.

        optimizer_class: `torch.optimizer.Optimizer`
            Any PyTorch optimizer should work. Additional arguments
            can be passed to this object via `**optimizer_kwargs`. The
            optimizer itself is built by `self.build_optimizer` when
            `fit` is called.

        l2_strength: float
            L2 regularization parameters for the optimizer. The default
            of 0 means no regularization, and larger values correspond
            to stronger regularization.

        gradient_accumulation_steps: int
            Controls how often the model parameters are updated during
            learning. For example, with `gradient_accumulation_steps=2`,
            the parameters are updated after every other batch. The primary
            use case for `gradient_accumulation_steps > 1` is where the
            model is very large, so only small batches of examples can be
            fit into memory. The updates based on these small batches can
            have high variance, so accumulating a few batches before
            updating can smooth the process out.

        max_grad_norm: None or float
            If not `None`, then `torch.nn.utils.clip_grad_norm_` is used
            to clip all the model parameters to within the range set
            by this value. This is a kind of brute-force way of keeping
            the parameter values from growing absurdly large or small.

        warm_start: bool
            If `False`, then repeated calls to `fit` will reset all the
            optimization settings: the model parameters, the optimizer,
            and the metadata we collect during optimization. If `True`,
            then calling `fit` twice with `max_iter=N` should be the same
            as calling fit once with `max_iter=N*2`.

        early_stopping: bool
            If `True`, then `validation_fraction` of the data given to
            `fit` are held out and used to assess the model after every
            epoch. The best scoring model is stored in an attribute
            `best_parameters`. If an improvement of at least `self.tol`
            isn't seen after `n_iter_no_change` iterations, then training
            stops and `self.model` is set to use `best_parameters`.

        validation_fraction: float
            Percentage of the data given to `fit` to hold out for use in
            early stopping. Ignored if `early_stopping=False`

        shuffle_train: bool
            Whether to shuffle the training data.

        n_iter_no_change: int
            Number of epochs used to control convergence and early
            stopping. Where `early_stopping=True`, training stops if an
            improvement of more than `self.tol` isn't seen after this
            many epochs. If `early_stopping=False`, then training stops
            if the epoch error doesn't drop by at least `self.tol` after
            this many epochs.

        tol: float
            Value used to control `early_stopping` and convergence.

        device: str or None
            Used to set the device on which the PyTorch computations will
            be done. If `device=None`, this will choose a CUDA device if
            one is available, else the CPU is used.

        display_progress: bool
            Whether to print optimization information incrementally to
            `sys.stderr` during training.

        **optimizer_kwargs: kwargs
            Any additional keywords given to the model will be passed to
            the optimizer -- see `self.build_optimizer`. The intent is to
            make it easy to tune these as hyperparameters will still
            allowing the user to specify just `optimizer_class` rather
            than setting up a full optimizer.

        Attributes
        ----------
        params: list
             All the keyword arguments are parameters and, with the
             exception of `display_progress`, their names are added to
             this list to support working with them using tools from
             `sklearn.model_selection`.

        """
        self.model = LIM
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.eta = eta
        self.optimizer_class = optimizer_class
        self.l2_strength = l2_strength
        self.gradient_accumulation_steps = max([gradient_accumulation_steps, 1])
        self.max_grad_norm = max_grad_norm
        self.warm_start = warm_start
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.shuffle_train = shuffle_train
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.first_run = True
        self.loss = nn.CrossEntropyLoss(reduction="mean")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.display_progress = display_progress
        self.optimizer_kwargs = optimizer_kwargs
        for k, v in self.optimizer_kwargs.items():
            setattr(self, k, v)
        self.params = [
            'batch_size',
            'max_iter',
            'eta',
            'optimizer_class',
            'l2_strength',
            'gradient_accumulation_steps',
            'max_grad_norm',
            'validation_fraction',
            'early_stopping',
            'n_iter_no_change',
            'warm_start',
            'tol']
        self.params += list(optimizer_kwargs.keys())

    def build_iit_dataset(self, base, base_y, iit_data):
        sources, IIT_y, intervention_ids = iit_data
        base = torch.FloatTensor(np.array(base))
        sources = [torch.FloatTensor(np.array(source)) for source in sources]
        sources = torch.reshape(
            torch.stack(sources, dim=1),
            (-1, len(sources),
            sources[0].shape[1]))

        intervention_ids = torch.FloatTensor(np.array(intervention_ids))

        base_y = np.array(base_y)
        self.classes_ = sorted(set(base_y))
        self.n_classes_ = len(self.classes_)
        class2index = dict(zip(self.classes_, range(self.n_classes_)))
        base_y = [class2index[label] for label in base_y]
        base_y = torch.tensor(base_y)

        IIT_y = np.array(IIT_y)
        IIT_y = [class2index[int(label)] for label in IIT_y]
        IIT_y = torch.tensor(IIT_y)

        dataset = torch.utils.data.TensorDataset(
            base, base_y, sources, IIT_y, intervention_ids)
        return dataset

    def build_dataset(self, base_x, base_y):
        base_x = torch.FloatTensor(np.array(base_x))

        base_y = np.array(base_y)
        self.classes_ = sorted(set(base_y))
        self.n_classes_ = len(self.classes_)
        class2index = dict(zip(self.classes_, range(self.n_classes_)))
        base_y = [class2index[label] for label in base_y]
        base_y = torch.tensor(base_y)

        dataset = torch.utils.data.TensorDataset(base_x, base_y)
        return dataset

    def build_optimizer(self):
        """
        Builds the optimizer. This function is called only when `fit`
        is called.

        Returns
        -------
        torch.optimizer.Optimizer

        """
        return self.optimizer_class(
            self.model.parameters(),
            lr=self.eta,
            weight_decay=self.l2_strength,
            **self.optimizer_kwargs)

    def process_batch(self,batch):
        return batch[0], batch[1]

    def process_IIT_batch(self,batch):
        return batch[2], batch[3], batch[4]

    def fit(self,
             base,
             base_labels,
             iit_data=None,
             intervention_ids_to_coords=None,
             device=None):
        """
        Generic optimization method.

        Parameters
        ----------
        *args: list of objects
            We assume that the final element of args give the labels
            and all the preceding elements give the system inputs.
            For regular supervised learning, this is like (X, y), but
            we allow for models that might use multiple data structures
            for their inputs.

        Attributes
        ----------
        model: nn.Module or subclass thereof
            Set by `build_graph`. If `warm_start=True`, then this is
            initialized only by the first call to `fit`.

        optimizer: torch.optimizer.Optimizer
            Set by `build_optimizer`. If `warm_start=True`, then this is
            initialized only by the first call to `fit`.

        errors: list of float
            List of errors. If `warm_start=True`, then this is
            initialized only by the first call to `fit`. Thus, where
            `max_iter=5`, if we call `fit` twice with `warm_start=True`,
            then `errors` will end up with 10 floats in it.

        validation_scores: list
            List of scores. This is filled only if `early_stopping=True`.
            If `warm_start=True`, then this is initialized only by the
            first call to `fit`. Thus, where `max_iter=5`, if we call
            `fit` twice with `warm_start=True`, then `validation_scores`
            will end up with 10 floats in it.

        no_improvement_count: int
            Used to control early stopping and convergence. These values
            are controlled by `_update_no_improvement_count_early_stopping`
            or `_update_no_improvement_count_errors`.  If `warm_start=True`,
            then this is initialized only by the first call to `fit`. Thus,
            in that situation, the values could accumulate across calls to
            `fit`.

        best_error: float
           Used to control convergence. Smaller is assumed to be better.
           If `warm_start=True`, then this is initialized only by the first
           call to `fit`. It will be reset by
           `_update_no_improvement_count_errors` depending on how the
           optimization is proceeding.

        best_score: float
           Used to control early stopping. If `warm_start=True`, then this
           is initialized only by the first call to `fit`. It will be reset
           by `_update_no_improvement_count_early_stopping` depending on how
           the optimization is proceeding. Important: we currently assume
           that larger scores are better. As a result, we will not get the
           correct results for, e.g., a scoring function based in
           `mean_squared_error`. See `self.score` for additional details.

        best_parameters: dict
            This is a PyTorch state dict. It is used if and only if
            `early_stopping=True`. In that case, it is updated whenever
            `best_score` is improved numerically. If the early stopping
            criteria are met, then `self.model` is reset to contain these
            parameters before `fit` exits.

        Returns
        -------
        self

        """
        if self.early_stopping:
            args, dev = self._build_validation_split(
                *args, validation_fraction=self.validation_fraction)

        # Dataset:
        if iit_data is not None:
            dataset = self.build_iit_dataset(base, base_labels, iit_data)
        else:
            dataset = self.build_dataset(base, base_labels)
        dataloader = self._build_dataloader(dataset, shuffle=self.shuffle_train)

        # Set up parameters needed to use the model. This is a separate
        # function to support using pretrained models for prediction,
        # where it might not be desirable to call `fit`.
        if self.first_run or not self.warm_start:
            self.initialize()
            self.first_run = False

        # Make sure the model is where we want it:
        self.model.to(self.device)

        self.model.train()
        self.optimizer.zero_grad()

        for iteration in range(1, self.max_iter+1):

            epoch_error = 0.0

            for batch_num, batch in enumerate(dataloader, start=1):

                batch = [x.to(self.device, non_blocking=True) for x in batch]
                base_batch, base_labels_batch  = self.process_batch(batch)
                batch_preds = self.model(base_batch)
                base_labels_batch = torch.squeeze(base_labels_batch)
                err = self.loss(batch_preds, base_labels_batch)
                if iit_data is not None:
                    sources_batch, iit_labels_batch, intervention_ids_batch \
                        = self.process_IIT_batch(batch)
                    batch_iit_preds = self.model.iit_forward(
                                    base_batch,
                                    sources_batch,
                                    intervention_ids_batch,
                                    intervention_ids_to_coords)
                    err += self.loss(batch_iit_preds, iit_labels_batch)
                if self.gradient_accumulation_steps > 1 and \
                  self.loss.reduction == "mean":
                    err /= self.gradient_accumulation_steps

                err.backward()

                epoch_error += err.item()

                if batch_num % self.gradient_accumulation_steps == 0 or \
                  batch_num == len(dataloader):
                    if self.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Stopping criteria:

            if self.early_stopping:
                self._update_no_improvement_count_early_stopping(*dev)
                if self.no_improvement_count > self.n_iter_no_change:
                    utils.progress_bar(
                        "Stopping after epoch {}. Validation score did "
                        "not improve by tol={} for more than {} epochs. "
                        "Final error is {}".format(iteration, self.tol,
                            self.n_iter_no_change, epoch_error),
                        verbose=self.display_progress)
                    break

            else:
                self._update_no_improvement_count_errors(epoch_error)
                if self.no_improvement_count > self.n_iter_no_change:
                    utils.progress_bar(
                        "Stopping after epoch {}. Training loss did "
                        "not improve more than tol={}. Final error "
                        "is {}.".format(iteration, self.tol, epoch_error),
                        verbose=self.display_progress)
                    break

            utils.progress_bar(
                "Finished epoch {} of {}; error is {}".format(
                    iteration, self.max_iter, epoch_error),
                verbose=self.display_progress)

        if self.early_stopping:
            self.model.load_state_dict(self.best_parameters)

        return self

    def initialize(self):
        """
        Method called by `fit` to establish core attributes. To use a
        pretrained model without calling `fit`, one can use this
        method.

        """
        # This device move has to happen before the optimizer is built:
        # https://pytorch.org/docs/master/optim.html#constructing-it
        self.optimizer = self.build_optimizer()
        self.model.to(self.device)
        self.errors = []
        self.validation_scores = []
        self.no_improvement_count = 0
        self.best_error = np.inf
        self.best_score = -np.inf
        self.best_parameters = None

    @staticmethod
    def _build_validation_split(*args, validation_fraction=0.2):
        """
        Split `*args` into train and dev portions for early stopping.
        We use `train_test_split`. For args of length N, then delivers
        N*2 objects, arranged as

        X1_train, X1_test, X2_train, X2_test, ..., y_train, y_test

        Parameters
        ----------
        *args: List of objects to split.

        validation_fraction: float
            Percentage of the examples to use for the dev portion. In
            `fit`, this is determined by `self.validation_fraction`.
            We give it as an argument here to facilitate unit testing.

        Returns
        -------
        Pair of tuples `train` and `dev`

        """
        if validation_fraction == 1.0:
            return args, args
        results = train_test_split(*args, test_size=validation_fraction)
        train = results[::2]
        dev = results[1::2]
        return train, dev

    def _build_dataloader(self, dataset, shuffle=True):
        """
        Internal method used to create a dataloader from a dataset.
        This is used by `fit` and `_predict`.

        Parameters
        ----------
        dataset: torch.utils.data.Dataset

        shuffle: bool
            When training, this is `True`. For prediction, this is
            crucially set to `False` so that the examples are not
            shuffled out of order with respect to labels that might
            be used for assessment.

        Returns
        -------
        torch.utils.data.DataLoader

        """
        if hasattr(dataset, "collate_fn"):
            collate_fn = dataset.collate_fn
        else:
            collate_fn = None
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=True,
            collate_fn=collate_fn)
        return dataloader

    def _update_no_improvement_count_early_stopping(self, *dev):
        """
        Internal method used by `fit` to control early stopping.
        The method uses `self.score(*dev)` for scoring and updates
        `self.validation_scores`, `self.no_improvement_count`,
        `self.best_score`, `self.best_parameters` as appropriate.

        """
        score = self.score(*dev)
        self.validation_scores.append(score)
        # If the score isn't at least `self.tol` better, increment:
        if score < (self.best_score + self.tol):
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0
        # If the current score is numerically better than all previous
        # scores, update the best parameters:
        if score > self.best_score:
            self.best_parameters = copy.deepcopy(self.model.state_dict())
            self.best_score = score
        self.model.train()

    def _update_no_improvement_count_errors(self, epoch_error):
        """
        Internal method used by `fit` to control convergence.
        The method uses `epoch_error`, `self.best_error`, and
        `self.tol` to make decisions, and it updates `self.errors`,
        `self.no_improvement_count`, and `self.best_error` as
        appropriate.

        """
        if epoch_error > (self.best_error - self.tol):
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0
        if epoch_error < self.best_error:
            self.best_error = epoch_error
        self.errors.append(epoch_error)

    def predict(self, X_base, device=None):
        """
        Internal method that subclasses are expected to use to define
        their own `predict` functions. The hope is that this method
        can do all the data organization and other details, allowing
        subclasses to have compact predict methods that just encode
        the core logic specific to them.

        Parameters
        ----------
        *args: system inputs

        device: str or None
            Allows the user to temporarily change the device used
            during prediction. This is useful if predictions require a
            lot of memory and so are better done on the CPU. After
            prediction is done, the model is returned to `self.device`.

        Returns
        -------
        The precise return value depends on the nature of the predictions.
        If the predictions have the same shape across all batches, then
        we return a single tensor concatenation of them. If the shape
        can vary across batches, as is common for sequence prediction,
        then we return a list of tensors of varying length.

        """
        device = self.device if device is None else torch.device(device)

        # Dataset:
        X_base = X_base.float().to(device)

        # Model:
        self.model.to(device)
        self.model.eval()

        with torch.no_grad():
            preds = self.model(X_base)

        # Make sure the model is back on the instance device:
        self.model.to(self.device)
        return preds.argmax(axis=1)

    def iit_predict(self,
                    base,
                    sources,
                    intervention_ids,
                    intervention_ids_to_coords,
                    device=None):
        """
        Internal method that subclasses are expected to use to define
        their own `predict` functions. The hope is that this method
        can do all the data organization and other details, allowing
        subclasses to have compact predict methods that just encode
        the core logic specific to them.

        Parameters
        ----------
        *args: system inputs

        device: str or None
            Allows the user to temporarily change the device used
            during prediction. This is useful if predictions require a
            lot of memory and so are better done on the CPU. After
            prediction is done, the model is returned to `self.device`.

        Returns
        -------
        The precise return value depends on the nature of the predictions.
        If the predictions have the same shape across all batches, then
        we return a single tensor concatenation of them. If the shape
        can vary across batches, as is common for sequence prediction,
        then we return a list of tensors of varying length.

        """
        device = self.device if device is None else torch.device(device)

        # Dataset:
        base = base.float().to(device)

        intervention_ids = intervention_ids.float().to(device)

        base_labels = [ 0 for _ in range(base.shape[0])]
        iit_labels = [ 0 for _ in range(base.shape[0])]

        dataset = self.build_iit_dataset(base, base_labels, (sources, iit_labels, intervention_ids))
        dataloader = self._build_dataloader(dataset, shuffle=False)

        # Model:
        self.model.set_device(device)
        self.model.eval()

        old_device = self.model.device
        self.model.device = device

        preds = None
        with torch.no_grad():
            for batch_num, batch in enumerate(dataloader, start=1):
                batch = [x.to(device, non_blocking=True) for x in batch]
                base_batch = batch[0]
                base_labels_batch = batch[1]
                sources_batch = batch[2]
                iit_labels_batch = batch[3]
                intervention_ids_batch = batch[4]
                batch_iit_preds = self.model.iit_forward(
                                base_batch,
                                sources_batch,
                                intervention_ids_batch,
                                intervention_ids_to_coords)
                if preds is None:
                    preds = batch_iit_preds
                else:
                    preds = torch.cat([preds, batch_iit_preds])

        # Make sure the model is back on the instance device:
        self.model.set_device(self.device)
        return preds.argmax(axis=1)


    def get_params(self, deep=True):
        params = self.params.copy()
        # Obligatorily add `vocab` so that sklearn passes it in when
        # creating new model instances during cross-validation:
        if hasattr(self, 'vocab'):
            params += ['vocab']
        return {p: getattr(self, p) for p in params}

    def set_params(self, **params):
        for key, val in params.items():
            if key not in self.params:
                raise ValueError(
                    "{} is not a parameter for {}. For the list of "
                    "available parameters, use `self.params`.".format(
                        key, self.__class__.__name__))
            else:
                setattr(self, key, val)
        return self

    def to_pickle(self, output_filename):
        """
        Serialize the entire class instance. Importantly, this is
        different from using the standard `torch.save` method:

        torch.save(self.model.state_dict(), output_filename)

        The above stores only the underlying model parameters. In
        contrast, the current method ensures that all of the model
        parameters are on the CPU and then stores the full instance.
        This is necessary to ensure that we retain all the information
        needed to read new examples, do additional training, make
        predictions, and so forth.

        Parameters
        ----------
        output_filename : str
            Full path for the output file.

        """
        self.model = self.model.cpu()
        with open(output_filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_pickle(src_filename):
        """
        Load an entire class instance onto the CPU. This also sets
        `self.warm_start=True` so that the loaded parameters are used
        if `fit` is called.

        Importantly, this is different from recommended PyTorch method:

        self.model.load_state_dict(torch.load(src_filename))

        We cannot reliably do this with new instances, because we need
        to see new examples in order to set some of the model
        dimensionalities and obtain information about what the class
        labels are. Thus, the current method loads an entire serialized
        class as created by `to_pickle`.

        The training and prediction code move the model parameters to
        `self.device`.

        Parameters
        ----------
        src_filename : str
            Full path to the serialized model file.

        """
        with open(src_filename, 'rb') as f:
            return pickle.load(f)

    def __repr__(self):
        param_str = ["{}={}".format(a, getattr(self, a)) for a in self.params]
        param_str = ",\n\t".join(param_str)
        return "{}(\n\t{})".format(self.__class__.__name__, param_str)




class BERTLIMTrainer(LIMTrainer):
    def __init__(self, bert, **kwargs):
        super().__init__(bert, **kwargs)

    def build_dataset(self, base_x, base_y):

        input, mask = base_x
        input = torch.stack(input)
        mask = torch.stack(mask)


        dataset = torch.utils.data.TensorDataset(input, mask, base_y)
        return dataset

    def build_iit_dataset(self, base, base_y, iit_data):

        base_y = torch.tensor(base_y)
        base_input, base_mask = base
        base_input = torch.stack(base_input)
        base_mask = torch.stack(base_mask)

        sources, IIT_y, intervention_ids = iit_data

        IIT_y = torch.tensor(IIT_y)

        if len(sources) == 1:
            sources_input, sources_mask = sources
        else:
            sources_input, sources_mask = zip(*sources)
        sources_input = [ torch.stack(input) for input in sources_input]
        sources_mask = [ torch.stack(mask) for mask in sources_mask]

        sources_input = torch.reshape(
            torch.stack(sources_input, dim=1),
            (-1, len(sources),
            sources_input[0].shape[1]))

        sources_mask = torch.reshape(
            torch.stack(sources_mask, dim=1),
            (-1, len(sources),
            sources_mask[0].shape[1]))

        intervention_ids = torch.FloatTensor(np.array(intervention_ids))

        dataset = torch.utils.data.TensorDataset(base_input,
                                                base_mask,
                                                base_y,
                                                sources_input,
                                                sources_mask,
                                                IIT_y,
                                                intervention_ids)
        return dataset

    def predict(self, X_base, device=None):
        """
        Internal method that subclasses are expected to use to define
        their own `predict` functions. The hope is that this method
        can do all the data organization and other details, allowing
        subclasses to have compact predict methods that just encode
        the core logic specific to them.

        Parameters
        ----------
        *args: system inputs

        device: str or None
            Allows the user to temporarily change the device used
            during prediction. This is useful if predictions require a
            lot of memory and so are better done on the CPU. After
            prediction is done, the model is returned to `self.device`.

        Returns
        -------
        The precise return value depends on the nature of the predictions.
        If the predictions have the same shape across all batches, then
        we return a single tensor concatenation of them. If the shape
        can vary across batches, as is common for sequence prediction,
        then we return a list of tensors of varying length.

        """
        device = self.device if device is None else torch.device(device)
        print(device)

        # Dataset:
        input, mask = X_base
        input = torch.stack(input, dim=0).to(device)
        mask = torch.stack(mask, dim=0).to(device)

        # Model:
        self.model.set_device(device)
        self.model.eval()

        with torch.no_grad():
            preds = self.model((input, mask))

        # Make sure the model is back on the instance device:
        self.model.set_device(self.device)
        return preds.argmax(axis=1)


    def process_batch(self,batch):
        return (batch[0], batch[1]), batch[2]

    def process_IIT_batch(self,batch):
        return (batch[3], batch[4]), batch[5], batch[6]


    def iit_predict(self,
                    base,
                    sources,
                    intervention_ids,
                    intervention_ids_to_coords,
                    device=None):
        """
        Internal method that subclasses are expected to use to define
        their own `predict` functions. The hope is that this method
        can do all the data organization and other details, allowing
        subclasses to have compact predict methods that just encode
        the core logic specific to them.

        Parameters
        ----------
        *args: system inputs

        device: str or None
            Allows the user to temporarily change the device used
            during prediction. This is useful if predictions require a
            lot of memory and so are better done on the CPU. After
            prediction is done, the model is returned to `self.device`.

        Returns
        -------
        The precise return value depends on the nature of the predictions.
        If the predictions have the same shape across all batches, then
        we return a single tensor concatenation of them. If the shape
        can vary across batches, as is common for sequence prediction,
        then we return a list of tensors of varying length.

        """
        device = self.device if device is None else torch.device(device)

        # Dataset:
        input_base, mask_base = base
        input_base = torch.stack(input_base, dim=0).to(device)
        mask_base = torch.stack(mask_base, dim=0).to(device)

        if len(sources) == 1:
            input_sources, mask_sources = sources[0]
            input_sources = torch.stack(input_sources, dim=0).to(device)
            mask_sources = torch.stack(mask_sources, dim=0).to(device)
        else:
            input_sources = []
            mask_sources = []
            for input_source, mask_source in sources:
                input_sources.append(torch.stack(input_source, dim=0))
                mask_sources.append(torch.stack(mask_source, dim=0))
            input_sources = torch.stack(input_sources, dim=1).to(device)
            mask_sources = torch.stack(mask_sources, dim=1).to(device)


        intervention_ids = intervention_ids.float().to(device)

        base_labels = [ 0 for _ in range(input_base.shape[0])]
        iit_labels = [ 0 for _ in range(input_base.shape[0])]


        # Model:
        self.model.set_device(device)
        self.model.eval()

        with torch.no_grad():
            preds = self.model.iit_forward((input_base, mask_base),
                                            (input_sources, mask_sources),
                                            intervention_ids,
                                            intervention_ids_to_coords)

        # Make sure the model is back on the instance device:
        self.model.set_device(self.device)
        return preds.argmax(axis=1)
