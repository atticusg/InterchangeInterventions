import random
import json
import copy
from typing import Optional
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from transformers import AutoTokenizer
import utils
from tqdm import tqdm
import wandb 

from flickr8k import evaluate_on_flickr8k
from food101 import evaluate_on_food101
from imagenet1k import evaluate_on_imagenet
from country211 import evaluate_on_country211
from cifar100 import evaluate_on_cifar100
from cosid import evaluate_on_cosid

CONCADIA_PATH = 'concadia.pt' 
CONCADIA_METADATA_PATH = 'wiki_split.json'
COLUMNS = ['filename', 'caption', 'description', 'image_features']
SEED = 42

def contrastive_loss(logits: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
    if labels is not None:
        return nn.functional.cross_entropy(logits, labels)
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_iit_loss(similarity: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
    # loss computes whether intervention affects final output
    iit_loss = contrastive_loss(similarity, labels)
    return iit_loss

class IITConcadiaDataset:
    def __init__(self, embed_func, split):
        self.embed_func = embed_func
        self.split = split
        concadia_data = torch.load(CONCADIA_PATH, map_location='cpu')
        concadia_df = pd.DataFrame(concadia_data, columns=COLUMNS).set_index('filename')
        with open(CONCADIA_METADATA_PATH) as f:
            concadia_metadata = json.load(f)['images']
        train_filenames = [im_data['filename'] for im_data in concadia_metadata if im_data['split'] == 'train']
        val_filenames = [im_data['filename'] for im_data in concadia_metadata if im_data['split'] == 'val']
        test_filenames = [im_data['filename'] for im_data in concadia_metadata if im_data['split'] == 'test']
        self.train_df = concadia_df.loc[train_filenames]
        self.val_df = concadia_df.loc[val_filenames]
        self.test_df = concadia_df.loc[test_filenames]

        if split == 'train':
            self.concadia_df = self.train_df
        elif split == 'val':
            self.concadia_df = self.val_df
        else:
            self.concadia_df = self.test_df

    def get_intervention(self, base, source):
        return 0

    def create_dataset(self, shuffle=True):
        data = []
        for i, row in self.concadia_df.iterrows():
            image_embeds = row['image_features']
            base, source = row['caption'], row['description']
            base_x, base_mask = self.embed_func(base)
            source_x, source_mask = self.embed_func(source)
            base_label = 0   # base label ignored during training
            intervention = self.get_intervention(base, source)
            IIT_label = 1    # 0 - prefer base input; 1 - prefer intervened input
            data.append((base_x, base_mask, base_label, source_x, source_mask, IIT_label, intervention, image_embeds))

            # repeat but with swapped base and source
            base, source = row['description'], row['caption']
            base_x, base_mask = self.embed_func(base)
            source_x, source_mask = self.embed_func(source)
            base_label = 0
            intervention = self.get_intervention(base, source)
            IIT_label = 0   # 0 - prefer base input; 1 - prefer intervened input
            data.append((base_x, base_mask, base_label, source_x, source_mask, IIT_label, intervention, image_embeds))

        if shuffle:
            # data.sort(key=lambda x: x[-2])
            # NOTE: JUST FOR FINETUNING
            # keep data in pairs, since the learning objective implicitly relies on this
            print('Shuffling data...')
            paired_data = [data[i:i+2] for i in range(0, len(data), 2)]
            random.shuffle(paired_data)
            data = [d for p in paired_data for d in p]

        base, base_mask, y, source, source_mask, IIT_y, interventions, image_embeds = zip(*data)
        self.base = base
        self.base_mask = base_mask
        self.source = source
        self.source_mask = source_mask
        self.y = np.array(y)
        self.IIT_y = np.array(IIT_y)
        self.interventions = np.array(interventions)
        self.image_embeds = image_embeds
        return (
            (self.base, self.base_mask, self.image_embeds), 
            self.y, 
            [(self.source,self.source_mask, self.image_embeds)], 
            self.IIT_y, 
            self.interventions
        )

def get_IIT_concadia_dataset(
    tokenizer_name,
    split="train",
    shuffle=True
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def encoding(X):
        input = [X]
        data = tokenizer(
            input,
            max_length=77,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        indices = data['input_ids']
        mask = data['attention_mask']
        return (indices, mask)
    
    dataset = IITConcadiaDataset(
        embed_func=encoding,
        split=split
    )
    
    X_base, y_base, X_sources,  y_IIT, interventions = dataset.create_dataset(shuffle=shuffle)
    y_base = torch.tensor(y_base)
    y_IIT = torch.tensor(y_IIT)
    interventions = torch.tensor(interventions)
    return X_base, y_base, X_sources, y_IIT, interventions

TEST_DATASET_IIT = get_IIT_concadia_dataset(
    split="test",
    tokenizer_name='openai/clip-vit-base-patch32',
    shuffle=False
)

def evaluate(clip_trainer, outdir, step, use_wandb=False):
    X_base_test = TEST_DATASET_IIT[0]
    
    print(f'Evaluating on concadia test ({step})...')
    base_preds_test = clip_trainer.predict(
        X_base_test
    )
    
    base_preds_test = base_preds_test.view((-1, 2))
    y_preds_test = base_preds_test.argmax(dim=-1).detach().cpu().numpy()
    bias = y_preds_test.mean()

    print(f"Evaluating on flickr8k ({step})...")
    flickr8k_eval = evaluate_on_flickr8k(clip_trainer.model)
    # print(f"Evaluating on food101 ({step})...")
    # food101_eval = evaluate_on_food101(clip_trainer.model)
    # print(f"Evaluating on imagenet ({step})...")
    # imagenet_eval = evaluate_on_imagenet(clip_trainer.model)
    # print(f"Evaluating on country211 ({step})...")
    # country211_eval = evaluate_on_country211(clip_trainer.model)
    # print(f"Evaluating on cifar100 ({step})...")
    # cifar100_eval = evaluate_on_cifar100(clip_trainer.model)

    with torch.no_grad():
        norms = [p.norm() for n, p in clip_trainer.model.named_parameters() if 'lora' in n]
        if len(norms) > 0:
            norm = sum(norms) / len(norms)
        else:
            norm = 0

    logs = {
        'eval_flickr8k': flickr8k_eval.correlation[0],
        # 'eval_food101': food101_eval,
        # 'eval_imagenet': imagenet_eval,
        # 'eval_country211': country211_eval,
        # 'eval_cifar100': cifar100_eval,
        'eval_accuracy': bias,
        'lora_norm': norm
    }

    # cosid_correlations, length_correlation = evaluate_on_cosid(clip_trainer.model)
    # for i, r in cosid_correlations.iterrows():
    #     logs[f'{r.Group} - {r.Aspect}'] = r.PearsonR

    if use_wandb:
        wandb.log(logs)
    else:
        with open(f'{outdir}/evaluation_{step}.csv', 'w+'):
            json.dump(logs)

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
            save_checkpoint_per_epoch=False,
            input_as_ids=False,
            class2index=None,
            seed=42,
            num_iter_per_val=1000,
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
        self.num_iter_per_val = num_iter_per_val

        # NOTE: INTRODUCE MSE LOSS FOR REGRESSION, CLIP LOSS FOR CONTRASTIVE LEARNING
        self.n_classes = self.model.n_classes
        if self.n_classes is None:
            self.loss = nn.CrossEntropyLoss()
        elif self.n_classes and self.n_classes == 1:
            self.loss = nn.MSELoss(reduction='mean')
        else:
            self.loss = nn.CrossEntropyLoss(reduction="mean")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.display_progress = display_progress
        self.save_checkpoint_per_epoch = save_checkpoint_per_epoch
        self.input_as_ids = input_as_ids
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
        self.class2index = class2index
        self.index2class = {}
        if class2index is not None:
            for k, v in class2index.items():
                self.index2class[v] = k
        self.seed = seed

    def build_iit_dataset(self, base, base_y, iit_data):
        sources, IIT_y, intervention_ids = iit_data
        if not self.input_as_ids:
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

        if self.class2index is not None:
            base_y = [self.class2index[label] for label in base_y]
        else:
            class2index = dict(zip(self.classes_, range(self.n_classes_)))
            base_y = [class2index[label] for label in base_y]
        base_y = torch.tensor(base_y)

        IIT_y = np.array(IIT_y)
        if self.class2index is not None:
            IIT_y = [self.class2index[int(label)] for label in IIT_y]
        else:
            class2index = dict(zip(self.classes_, range(self.n_classes_)))
            IIT_y = [class2index[int(label)] for label in IIT_y]
        IIT_y = torch.tensor(IIT_y)

        dataset = torch.utils.data.TensorDataset(
            base, base_y, sources, IIT_y, intervention_ids)
        return dataset

    def build_dataset(self, base_x, base_y):
        if not self.input_as_ids:
            base_x = torch.FloatTensor(np.array(base_x))

        base_y = np.array(base_y)
        self.classes_ = sorted(set(base_y))
        self.n_classes_ = len(self.classes_)

        if self.class2index is not None:
            base_y = [self.class2index[label] for label in base_y]
        else:
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
             base_val=None,
             base_labels_val=None,
             iit_data_val=None,
             intervention_ids_to_coords=None,
             device=None,
             outdir=None,
             use_wandb=False,
             multitask=-1.
           ):
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
        # if self.early_stopping:
            # args, dev = self._build_validation_split(
            #     *args, validation_fraction=self.validation_fraction)

        # build validation dataloader if provided validation data
        if iit_data_val is not None:
            dataset_val = self.build_iit_dataset(base_val, base_labels_val, iit_data_val)
            dataloader_val = self._build_dataloader(dataset_val, shuffle=False)
        elif base_val is not None:
            dataset_val = self.build_dataset(base_val, base_labels_val)
            dataloader_val = self._build_dataloader(dataset_val, shuffle=False)
        else:
            dataloader_val = None
        
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
        self.model.set_device(self.device)

        self.model.train()
        self.optimizer.zero_grad()

        # for iteration in range(1, self.max_iter+1):
        # epoch_error = 0.0
        
        epoch = 0
        while epoch < self.max_iter:
            epoch += 1
            with tqdm(dataloader, desc=f'Epoch {epoch}') as pbar:
                for batch_num, batch in enumerate(pbar):
                    batch = [x.to(self.device) for x in batch]
                    base_batch, base_labels_batch  = self.process_batch(batch)

                    # NOTE: AMIR ADDITION - LEARN INTERVENTION BY CALLING ON FORWARD WITH INTERVENTION
                    if self.model.learn_intervention_vector:
                        batch_preds = self.model.forward_with_intervention(
                            base_batch, intervention_ids_to_coords
                        )
                    else:
                        batch_preds = self.model(base_batch)

                    base_labels_batch = torch.squeeze(base_labels_batch)

                    # NOTE: Converting int to long for CE, adding MSE loss case
                    # if learning intervention vector, do not rely on labels during training
                    if self.model.learn_intervention_vector:
                        pass
                    elif self.n_classes is None:
                        # evaluate non-IIT loss by comparing descriptions and captions
                        # base labels can be read by the index of the first logit (specifically for our implementation of CLIP)
                        if iit_data is None:
                            base_labels_batch = base_labels_batch.view(-1, 2)[:, 0]
                            base_preds = batch_preds.diag().view(base_labels_batch.size(0), 2)
                            err = self.loss(base_preds, base_labels_batch.to(torch.long))

                    elif self.n_classes == 1:
                        err = self.loss(batch_preds.squeeze(), base_labels_batch.to(torch.float).squeeze())
                    else:
                        err = self.loss(batch_preds, base_labels_batch.to(torch.long))

                    if iit_data is not None:
                        sources_batch, iit_labels_batch, intervention_ids_batch \
                            = self.process_IIT_batch(batch)

                        if self.model.learn_intervention_vector:
                            batch_iit_preds = self.model.forward_with_intervention(
                                sources_batch, intervention_ids_to_coords
                            )
                        # elif self.model.multitask_objective:
                        #     batch_iit_preds, multitask_preds = self.model.iit_forward(
                        #         base_batch,
                        #         sources_batch,
                        #         intervention_ids_batch,
                        #         intervention_ids_to_coords
                        #     )
                        else:
                            batch_iit_preds = self.model.iit_forward(
                                base_batch,
                                sources_batch,
                                intervention_ids_batch,
                                intervention_ids_to_coords
                            )
                        # NOTE: Converting int to long for CE, adding MSE loss case
                        if self.n_classes is None:
                            base_preds = batch_preds.diag()
                            iit_preds = batch_iit_preds.diag()
                            preds = torch.stack((base_preds, iit_preds)).t()
                            err = self.loss(preds, iit_labels_batch.to(torch.long))

                            # for multitask objective, run finetuning objective
                            # but on source vs. base inputs (i.e., input-level intervention)
                            # (can't directly apply finetuning objective like above, b/c it implicitly
                            # relies on captions & descriptions to be consecutive batch inputs)
                            if multitask >= 0:
                                source_preds = self.model(sources_batch).diag()
                                behavioral_preds = torch.stack((base_preds, source_preds)).t()
                                behavioral_err = self.loss(behavioral_preds, iit_labels_batch.to(torch.long))
                                err = multitask * err + (1 - multitask) * behavioral_err

                            # if self.model.multitask_objective:
                            #     # compute multitask objective as whether we predicted the correct labels
                            #     multitask_err = self.loss(multitask_preds, base_labels_batch.to(torch.long))
                            #     err = err + multitask_err
                        elif self.model.learn_intervention_vector:
                            err = self.loss(batch_preds.squeeze(), batch_iit_preds.squeeze())

                        elif self.n_classes == 1:
                            err += self.loss(batch_iit_preds.squeeze(), iit_labels_batch.to(torch.float).squeeze())
                        else:
                            err += self.loss(batch_iit_preds, iit_labels_batch.to(torch.long))
                            
                    if self.gradient_accumulation_steps > 1 and \
                    self.loss.reduction == "mean":
                        err /= self.gradient_accumulation_steps

                    self.errors.append(err.item())

                    pbar.set_postfix({'loss': err.item()})

                    if use_wandb:
                        wandb.log({"train_loss": err.item()})

                    err.backward()

                    # if self.early_stopping:
                    #     self._update_no_improvement_count_errors(err.item())
                    #     if self.no_improvement_count > self.n_iter_no_change:
                    #         utils.progress_bar(
                    #             "Stopping after epoch {}. Training loss did "
                    #             "not improve more than tol={}. Final error "
                    #             "is {}.".format(iteration, self.tol, err.item()),
                    #             verbose=self.display_progress)
                    #         break

                    if batch_num % self.gradient_accumulation_steps == 0 or \
                        batch_num == len(dataloader):
                        if self.max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    # Stopping criteria:
                    if dataloader_val is not None and batch_num % self.num_iter_per_val == 0:
                        self._update_no_improvement_count_early_stopping(
                            dataloader=dataloader_val,
                            is_iit=(iit_data_val is not None),
                            intervention_ids_to_coords=intervention_ids_to_coords,
                            use_wandb=use_wandb
                        )
                        if self.early_stopping and self.no_improvement_count >= self.n_iter_no_change:
                            # utils.progress_bar(
                            #     "Stopping after epoch {}. Validation score did "
                            #     "not improve by tol={} for more than {} epochs. "
                            #     "Final validation error is {}".format(iteration, self.tol, self.n_iter_no_change, self.best_score),
                            #     verbose=self.display_progress)
                            print('Final error:', self.best_score)
                            utils.progress_bar(
                                f'Stopping after {batch_num} batches. Validation score '
                                f'did not improve by tol={self.tol} for more than {self.n_iter_no_change} epochs.'
                                f'Final validation error is {self.best_score}.',
                                verbose=self.display_progress
                            )
                            # end training
                            running = False
                            break
                    
                    num_iter_per_step = self.num_iter_per_val # (self.num_iter_per_val // 5)
                    if batch_num % num_iter_per_step == 0:
                        n = (epoch - 1) * (len(pbar) // num_iter_per_step) + batch_num // num_iter_per_step
                        evaluate(self, outdir, n, use_wandb=use_wandb)

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
        self.best_score = np.inf
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
            pin_memory=False,
            collate_fn=collate_fn)
        return dataloader

    def _update_no_improvement_count_early_stopping(
        self, 
        dataloader, 
        is_iit,
        intervention_ids_to_coords,
        use_wandb=False
    ):
        """
        Internal method used by `fit` to control early stopping.
        The method uses `self.score(*dev)` for scoring and updates
        `self.validation_scores`, `self.no_improvement_count`,
        `self.best_score`, `self.best_parameters` as appropriate.

        """
        self.model.eval()
        score = 0.

        with torch.no_grad():
            with tqdm(dataloader, desc='Validating...') as pbar:
                for batch_num, batch in enumerate(pbar, start=1):
                    batch = [x.to(self.device) for x in batch]
                    base_batch, base_labels_batch  = self.process_batch(batch)

                    batch_preds = self.model(base_batch)

                    base_labels_batch = torch.squeeze(base_labels_batch)

                    # NOTE: Converting int to long for CE, adding MSE loss case
                    # if learning intervention vector, do not rely on labels during training
                    if self.n_classes is None:
                        # evaluate non-IIT loss by comparing descriptions and captions
                        # base labels can be read by the index of the first logit (specifically for our implementation of CLIP)
                        if not is_iit:
                            # NOTE: this relies on the IIT dataset pairing description + caption pairs right next to each other!
                            base_labels_batch = base_labels_batch.view(-1, 2)[:, 0]
                            base_preds = batch_preds.diag().view(base_labels_batch.size(0), 2)
                            err = self.loss(base_preds, base_labels_batch.to(torch.long))

                    elif self.n_classes == 1:
                        err = self.loss(batch_preds.squeeze(), base_labels_batch.to(torch.float).squeeze())
                    else:
                        err = self.loss(batch_preds, base_labels_batch.to(torch.long))

                    if is_iit:
                        sources_batch, iit_labels_batch, intervention_ids_batch \
                            = self.process_IIT_batch(batch)
                        batch_iit_preds = self.model.iit_forward(
                            base_batch,
                            sources_batch,
                            intervention_ids_batch,
                            intervention_ids_to_coords
                        )
                        # NOTE: Converting int to long for CE, adding MSE loss case
                        if self.n_classes is None:
                            base_preds = batch_preds.diag()
                            iit_preds = batch_iit_preds.diag()
                            preds = torch.stack((base_preds, iit_preds)).t()
                            err = self.loss(preds, iit_labels_batch.to(torch.long))
                        elif self.n_classes == 1:
                            err += self.loss(batch_iit_preds.squeeze(), iit_labels_batch.to(torch.float).squeeze())
                        else:
                            err += self.loss(batch_iit_preds, iit_labels_batch.to(torch.long))

                    score += err.item()
                    pbar.set_postfix({'loss': err.item(), 'epoch_loss': score})

        self.model.train()

        score = score / len(dataloader)

        if use_wandb:
            wandb.log({"eval_loss": score})

        self.validation_scores.append(score)

        if score > (self.best_score - self.tol):
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0

        if score < self.best_score:
            self.best_score = score
            self.best_parameters = self.model.state_dict()

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

        if not self.input_as_ids:
            # Dataset:
            X_base = X_base.float()
        X_base = X_base.to(device)

        # Model:
        self.model.to(device)
        self.model.eval()

        with torch.no_grad():
            preds = self.model(X_base)

        # Make sure the model is back on the instance device:
        self.model.to(self.device)

        # NOTE: Adding result for regression
        # (no need to take argmax)
        if self.n_classes == 1:
            return preds.squeeze()

        if self.class2index is not None:
            preds = preds.argmax(axis=1)
            preds = np.array(preds)
            preds = [self.index2class[label] for label in preds]
            preds = torch.tensor(preds)
        else:
            preds = preds.argmax(axis=1)

        return preds

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
        base = base.to(device)
        sources = [source.to(device) for source in sources]

        intervention_ids = intervention_ids.float().to(device)

        if self.class2index is None:
            base_labels = [ 0 for _ in range(base.shape[0])]
            iit_labels = [ 0 for _ in range(base.shape[0])]
        else:
            base_labels = [ 2 for _ in range(base.shape[0])]
            iit_labels = [ 2 for _ in range(base.shape[0])]

        dataset = self.build_iit_dataset(base, base_labels, (sources, iit_labels, intervention_ids))
        dataloader = self._build_dataloader(dataset, shuffle=False)

        # Model:
        self.model.set_device(device)

        old_device = self.model.device
        self.model.device = device

        self.model.eval()

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
        self.model.device = old_device

        # NOTE: Adding result for regression
        # (no need to take argmax)
        if self.n_classes == 1:
            return preds.squeeze()

        if self.class2index is not None:
            preds = preds.argmax(axis=1)
            preds = np.array(preds)
            preds = [self.index2class[label] for label in preds]
            preds = torch.tensor(preds)
        else:
            preds = preds.argmax(axis=1)

        return preds

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

        base_input, base_mask = base
        base_input = torch.stack(base_input)
        base_mask = torch.stack(base_mask)

        sources, IIT_y, intervention_ids = iit_data

        if len(sources) == 1:
            sources_input, sources_mask = sources[0]
            sources_input = [sources_input]
            sources_mask = [sources_mask]
        else:
            sources_input, sources_mask = zip(*sources)
        sources_input = [ torch.stack(input) for input in sources_input]
        sources_mask = [ torch.stack(mask) for mask in sources_mask]

        sources_input = torch.reshape(
            torch.stack(sources_input, dim=1),
            (-1, len(sources),
            base_input.shape[-1]))

        sources_mask = torch.reshape(
            torch.stack(sources_mask, dim=1),
            (-1, len(sources),
            base_input.shape[-1]))

        if isinstance(intervention_ids, list):
            intervention_ids = torch.FloatTensor(np.array(intervention_ids))

        dataset = torch.utils.data.TensorDataset(base_input,
                                                base_mask,
                                                base_y,
                                                sources_input,
                                                sources_mask,
                                                IIT_y,
                                                intervention_ids)
        return dataset

    def process_batch(self, batch, device=None):
        if device == None:
            device = self.device
        # cast tensors to trainer device
        return (
            (batch[0].squeeze().to(device), batch[1].squeeze().to(device)),
            batch[2].to(device)
        )

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
        y_base = torch.tensor([0 for _ in range(len(X_base[0]))])
        dataset = self.build_dataset(X_base, y_base)
        dataloader = self._build_dataloader(dataset, shuffle=False)
        # Dataset:

        # Model:
        self.model.set_device(device)
        self.model.eval()
        preds = None

        with torch.no_grad():
            for batch_num, batch in enumerate(dataloader, start=1):
                batch = [x.to(device, non_blocking=True) for x in batch]
                base_batch, base_labels_batch = self.process_batch(batch, device=device)
                batch_preds = self.model.forward(
                                base_batch)
                if preds is None:
                    preds = batch_preds
                else:
                    preds = torch.cat([preds, batch_preds])
        # Make sure the model is back on the instance device:
        self.model.set_device(self.device)

        # NOTE: Adding result for regression
        # (no need to take argmax)
        if self.n_classes == 1:
            return preds.squeeze()

        return preds.argmax(axis=1)

    def predict_with_intervention(self, X_base, gets, intervention, variable=0, device=None):
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
        y_base = torch.tensor([0 for _ in range(len(X_base[0]))])
        dataset = self.build_dataset(X_base, y_base)
        dataloader = self._build_dataloader(dataset, shuffle=False)
        # Dataset:

        # Model:
        self.model.set_device(device)
        self.model.eval()
        preds = None

        sets = copy.deepcopy(gets)
        sets[variable]['intervention'] = intervention.repeat((self.batch_size, 1))

        with torch.no_grad():
            for batch_num, batch in enumerate(dataloader, start=1):
                batch = [x.to(device, non_blocking=True) for x in batch]
                base_batch, base_labels_batch = self.process_batch(batch, device=device)

                # account for last batch, which might have smaller size
                if base_batch[0].size(0) < self.batch_size:
                    sets[0]['intervention'] = intervention.repeat((base_batch[0].size(0), 1))

                # apply intervention
                handlers = self.model._gets_sets(gets=None, sets=sets)
                batch_preds = self.model.forward(
                                base_batch)
                # clean up intervention hooks
                for handler in handlers:
                    handler.remove()

                if preds is None:
                    preds = batch_preds
                else:
                    preds = torch.cat([preds, batch_preds])
        # Make sure the model is back on the instance device:
        self.model.set_device(self.device)

        # NOTE: Adding result for regression
        # (no need to take argmax)
        if self.n_classes == 1:
            return preds.squeeze()

        return preds.argmax(axis=1)

    def process_IIT_batch(self,batch):
        # cast tensors to trainer device
        return (
            (batch[3].squeeze().to(self.device), batch[4].squeeze().to(self.device)),
            batch[5].to(self.device),
            batch[6].to(self.device)
        )

    def iit_predict(
        self,
        base,
        sources,
        intervention_ids,
        intervention_ids_to_coords,
        device=None
    ):
        """
        NOTE: rewrote original IIT predict (seemed to have some problems with
        vector shapes). This one hopefully better matches the format of `iit_predict`
        written in the more general `LIMTrainer` class.
        (in fact, should be identical if we ensure that `base` and `sources` are cast
        to tensors)
        """
        device = self.device if device is None else torch.device(device)

        # Dataset:
        # base = base.float().to(device)
        # sources = [source.to(device) for source in sources]

        intervention_ids = intervention_ids.float().to(device)

        # base_labels = [ 0 for _ in range(base.shape[0])]
        # iit_labels = [ 0 for _ in range(base.shape[0])]
        base_labels = iit_labels = torch.tensor([0] * len(base[0]))

        iit_data = (sources, iit_labels, intervention_ids)
        dataset = self.build_iit_dataset(base, base_labels, iit_data)
        dataloader = self._build_dataloader(dataset, shuffle=False)

        # Model:
        self.model.set_device(device)

        old_device = self.model.device
        self.model.device = device
        # Temporary fix: set self.device to be specified device, then set back
        # (better solution is to add device as an optional parameter to process batch)
        self.device = device

        self.model.eval()

        preds = None
        with torch.no_grad():
            for batch_num, batch in enumerate(dataloader, start=1):
                batch = [x.to(device, non_blocking=True) for x in batch]
                base_batch, base_labels_batch = self.process_batch(batch)
                # base_batch = batch[0]
                # base_labels_batch = batch[1]
                sources_batch, iit_labels_batch, intervention_ids_batch = self.process_IIT_batch(batch)
                # sources_batch = batch[2]
                # iit_labels_batch = batch[3]
                # intervention_ids_batch = batch[4]
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
        self.device = old_device
        self.model.set_device(self.device)
        self.model.device = old_device

        # NOTE: Adding result for regression
        # (no need to take argmax)
        if self.n_classes == 1:
            return preds.squeeze()

        return preds.argmax(axis=1)


class CLIPLIMTrainer(LIMTrainer):
    def __init__(self, lim_clip, **kwargs):
        super().__init__(lim_clip, **kwargs)

    def build_dataset(self, base_x, base_y):

        input, mask, image_embeds = base_x
        input = torch.stack(input)
        mask = torch.stack(mask)
        image_embeds = torch.stack(image_embeds)

        dataset = torch.utils.data.TensorDataset(input, mask, image_embeds, base_y)
        return dataset

    def build_iit_dataset(self, base, base_y, iit_data):

        base_input, base_mask, base_image_embeds = base
        base_input = torch.stack(base_input)
        base_mask = torch.stack(base_mask)
        base_image_embeds = torch.stack(base_image_embeds)

        sources, IIT_y, intervention_ids = iit_data

        if len(sources) == 1:
            sources_input, sources_mask, sources_image_embeds = sources[0]
            sources_input = [sources_input]
            sources_mask = [sources_mask]
            sources_image_embeds = [sources_image_embeds]
        else:
            sources_input, sources_mask, sources_image_embeds = zip(*sources)
        sources_input = [ torch.stack(input) for input in sources_input]
        sources_mask = [ torch.stack(mask) for mask in sources_mask]
        sources_image_embeds = [ torch.stack(image_embed) for image_embed in sources_image_embeds]

        sources_input = torch.reshape(
            torch.stack(sources_input, dim=1),
            (-1, len(sources),
            base_input.shape[-1]))

        sources_mask = torch.reshape(
            torch.stack(sources_mask, dim=1),
            (-1, len(sources),
            base_input.shape[-1]))

        sources_image_embeds = torch.reshape(
            torch.stack(sources_image_embeds, dim=1),
            (-1, len(sources),
            base_image_embeds.shape[-1]))

        if isinstance(intervention_ids, list):
            intervention_ids = torch.FloatTensor(np.array(intervention_ids))

        dataset = torch.utils.data.TensorDataset(base_input,
                                                base_mask,
                                                base_image_embeds,
                                                base_y,
                                                sources_input,
                                                sources_mask,
                                                sources_image_embeds,
                                                IIT_y,
                                                intervention_ids)
        return dataset

    def process_batch(self, batch, device=None):
        if device == None:
            device = self.device
        # cast tensors to trainer device
        return (
            (batch[0].squeeze().to(device), batch[1].squeeze().to(device), batch[2].squeeze(0).to(device)),
            batch[3].to(device)
        )

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
        y_base = torch.tensor([0 for _ in range(len(X_base[0]))])
        dataset = self.build_dataset(X_base, y_base)
        dataloader = self._build_dataloader(dataset, shuffle=False)
        # Dataset:

        # Model:
        self.model.set_device(device)
        self.model.eval()
        preds = None

        with torch.no_grad():
            for batch_num, batch in enumerate(tqdm(dataloader), start=1):
                batch = [x.to(device, non_blocking=True) for x in batch]
                base_batch, base_labels_batch = self.process_batch(batch, device=device)

                batch_preds = self.model.forward(
                    base_batch
                )

                # FOR CLIP ONLY: we are only concerned with diagonal entries (right image to right description)
                batch_preds = batch_preds.diag()

                if preds is None:
                    preds = batch_preds
                else:
                    preds = torch.cat([preds, batch_preds])
        # Make sure the model is back on the instance device:
        self.model.set_device(self.device)

        # NOTE: Adding result for regression
        # (no need to take argmax)
        if self.n_classes is None or self.n_classes == 1:
            return preds.squeeze()

        return preds.argmax(axis=1)

    def predict_with_intervention(self, X_base, gets, intervention, variable=0, device=None):
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
        y_base = torch.tensor([0 for _ in range(len(X_base[0]))])
        dataset = self.build_dataset(X_base, y_base)
        dataloader = self._build_dataloader(dataset, shuffle=False)
        # Dataset:

        # Model:
        self.model.set_device(device)
        self.model.eval()
        preds = None

        sets = copy.deepcopy(gets)
        sets[variable]['intervention'] = intervention.repeat((self.batch_size, 1))

        with torch.no_grad():
            for batch_num, batch in enumerate(dataloader, start=1):
                batch = [x.to(device, non_blocking=True) for x in batch]
                base_batch, base_labels_batch = self.process_batch(batch, device=device)

                # account for last batch, which might have smaller size
                if base_batch[0].size(0) < self.batch_size:
                    sets[0]['intervention'] = intervention.repeat((base_batch[0].size(0), 1))

                # apply intervention
                handlers = self.model._gets_sets(gets=None, sets=sets)
                batch_preds = self.model.forward(
                                base_batch)
                # clean up intervention hooks
                for handler in handlers:
                    handler.remove()

                if preds is None:
                    preds = batch_preds
                else:
                    preds = torch.cat([preds, batch_preds])
        # Make sure the model is back on the instance device:
        self.model.set_device(self.device)

        # NOTE: Adding result for regression
        # (no need to take argmax)
        if self.n_classes == 1:
            return preds.squeeze()

        return preds.argmax(axis=1)

    def process_IIT_batch(self, batch):
        # cast tensors to trainer device
        return (
            (batch[4].squeeze().to(self.device), batch[5].squeeze().to(self.device), batch[6].squeeze(0).to(self.device)),
            batch[7].to(self.device),
            batch[8].to(self.device)
        )

    def iit_predict(
        self,
        base,
        sources,
        intervention_ids,
        intervention_ids_to_coords,
        device=None
    ):
        """
        NOTE: rewrote original IIT predict (seemed to have some problems with
        vector shapes). This one hopefully better matches the format of `iit_predict`
        written in the more general `LIMTrainer` class.
        (in fact, should be identical if we ensure that `base` and `sources` are cast
        to tensors)
        """
        device = self.device if device is None else torch.device(device)

        # Dataset:
        # base = base.float().to(device)
        # sources = [source.to(device) for source in sources]

        intervention_ids = intervention_ids.float().to(device)

        # base_labels = [ 0 for _ in range(base.shape[0])]
        # iit_labels = [ 0 for _ in range(base.shape[0])]
        base_labels = iit_labels = torch.tensor([0] * len(base[0]))

        iit_data = (sources, iit_labels, intervention_ids)
        dataset = self.build_iit_dataset(base, base_labels, iit_data)
        dataloader = self._build_dataloader(dataset, shuffle=False)

        # Model:
        self.model.set_device(device)

        old_device = self.model.device
        self.model.device = device
        # Temporary fix: set self.device to be specified device, then set back
        # (better solution is to add device as an optional parameter to process batch)
        self.device = device

        self.model.eval()

        preds = None
        with torch.no_grad():
            for batch_num, batch in enumerate(tqdm(dataloader), start=1):
                batch = [x.to(device, non_blocking=True) for x in batch]
                base_batch, base_labels_batch = self.process_batch(batch)

                # base_batch = batch[0]
                # base_labels_batch = batch[1]
                sources_batch, iit_labels_batch, intervention_ids_batch = self.process_IIT_batch(batch)
                # sources_batch = batch[2]
                # iit_labels_batch = batch[3]
                # intervention_ids_batch = batch[4]
                batch_iit_preds = self.model.iit_forward(
                    base_batch,
                    sources_batch,
                    intervention_ids_batch,
                    intervention_ids_to_coords
                )

                # FOR CLIP ONLY: we are only concerned with diagonal entries (right image to right description)
                batch_iit_preds = batch_iit_preds.diag()
                if preds is None:
                    preds = batch_iit_preds
                else:
                    preds = torch.cat([preds, batch_iit_preds])

        # Make sure the model is back on the instance device:
        self.device = old_device
        self.model.set_device(self.device)
        self.model.device = old_device

        # NOTE: Adding result for regression and for CLIP
        # (no need to take argmax)
        if self.n_classes is None or self.n_classes == 1:
            return preds.squeeze()

        return preds.argmax(axis=1)
