import os, sys, json, random, argparse, itertools
from functools import reduce

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

import torch
from torchvision.datasets import CIFAR100

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from transformers import AutoTokenizer, CLIPModel

from peft import LoraConfig, get_peft_model

from LIM_clip import LIMClipTextModel
from trainer import CLIPLIMTrainer

from utils import fix_random_seeds

from flickr8k import evaluate_on_flickr8k
from food101 import evaluate_on_food101
from imagenet1k import evaluate_on_imagenet
from sst2 import evaluate_on_sst2
from country211 import evaluate_on_country211
from cifar100 import evaluate_on_cifar100
from ig import evaluate_on_ig
from cosid import evaluate_on_cosid

CONCADIA_PATH = 'concadia.pt' 
CONCADIA_METADATA_PATH = 'wiki_split.json'
COLUMNS = ['filename', 'caption', 'description', 'image_features']

VAR = 0
SEED = 42

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
        return VAR

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

class ConcadiaDataset:
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

    def create_dataset(self, size=None, shuffle=False):
        if size is not None:
            df = self.concadia_df.sample(size, replace=False, random_state=SEED)
        else:
            df = self.concadia_df

        data = []
        for i, row in df.iterrows():
            image_embeds = row['image_features']
            caption, description = row['caption'], row['description']
            base_x, base_mask = self.embed_func(caption)
            base_label = 1 # specify that we prefer description to caption
            data.append((base_x, base_mask, base_label, image_embeds))

            base_x, base_mask = self.embed_func(description)
            base_label = 1 # this is ignored??
            data.append((base_x, base_mask, base_label, image_embeds))

        if shuffle:
            # NOTE: JUST FOR FINETUNING
            # keep data in pairs, since the learning objective implicitly relies on this
            print('Shuffling data...')
            paired_data = [data[i:i+2] for i in range(0, len(data), 2)]
            random.shuffle(paired_data)
            data = [d for p in paired_data for d in p]
        
        base, base_mask, y, image_embeds = zip(*data)
        self.base = base
        self.base_mask = base_mask
        self.y = np.array(y)
        self.image_embeds = image_embeds
        return (
            (self.base, self.base_mask, self.image_embeds), 
            self.y
        )
    
def get_concadia_dataset(
    tokenizer_name,
    size=None,
    split="train",
    shuffle=False
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
    
    dataset = ConcadiaDataset(
        embed_func=encoding,
        split=split
    )
    
    X_base, y_base = dataset.create_dataset(size=size, shuffle=shuffle)
    y_base = torch.tensor(y_base)
    return X_base, y_base


def main(args):
    fix_random_seeds(args.seed)

    wandb_run = wandb.init(
        # Set the project where this run will be logged
        project="concadia-iit-emnlp",
        group='bft',
        config={
            'method': 'bft',
            'l2': args.l2,
            'lora': args.lora,
            'rank': args.rank,
            'lora_dropout': args.lora_dropout,
            'seed': args.seed
        }
    )

    print('Loading pre-trained CLIP...')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    iit_layer = 10
    hidden_dim_per_concept = 128
    model_dim = 512

    intervention_ids_to_coords = {
        VAR: [{"layer":iit_layer, "start":0, "end":hidden_dim_per_concept}]
    }

    lim_clip = LIMClipTextModel(
        clip, 
        max_length=77, 
        device=device, 
        target_layers=[iit_layer], 
        target_dims={ "start": 0, "end": model_dim },
        static_search=True
    )
    
    lim_clip.set_analysis_mode(False)

    if args.lora:
        target_modules = ['q_proj', 'v_proj']
        if args.all_attention:
            target_modules += ['k_proj', 'out_proj']
        elif args.all_layers:
            target_modules += ['k_proj', 'out_proj', 'fc1', 'fc2']
            
        config = LoraConfig(
            r=args.rank,    
            lora_alpha=args.rank,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )

        lim_clip = get_peft_model(lim_clip, config)

    training_parameters={
        'warm_start': False, 
        'max_iter': args.max_iter, 
        'batch_size': 12,
        'l2_strength': args.l2,
        'shuffle_train': False, 
        'eta': args.lr,
        'device': device,
        # early stopping parameters
        'tol': args.tol,
        'early_stopping': args.early_stopping,
        'n_iter_no_change': args.n_iter_no_change,
        'num_iter_per_val': args.num_iter_per_val   
    }

    clip_trainer = CLIPLIMTrainer(
        lim_clip,
        **training_parameters
    )
    
    print('Loading training data...')
    train_dataset = get_concadia_dataset(
        split="train",
        size=args.train_size,
        tokenizer_name='openai/clip-vit-base-patch32',
        shuffle=True
    )
    X_base_train, y_base_train = train_dataset
    print(f'\t\tTraining data loaded: {len(y_base_train)} datapoints.')

    print('Loading validation data...')
    val_dataset = get_concadia_dataset(
        split="val",
        size=args.val_size,
        tokenizer_name='openai/clip-vit-base-patch32',
        shuffle=False
    )
    X_base_val, y_base_val = val_dataset
    print(f'\t\tValidation data loaded: {len(y_base_val)} datapoints.')

    argument_strs = [
        f'{k}={v}' for k, v in vars(args).items()
        if k not in ['out_dir', 'early_stopping', 'max_iter', 'train_size', 'val_size']
    ]
    outdir = f'{args.out_dir}/' + reduce(lambda s, t: s + '_' + t, argument_strs)
    os.makedirs(outdir, exist_ok=True)
    
    _ = clip_trainer.fit(
        X_base_train, 
        y_base_train, 
        base_val=X_base_val,
        base_labels_val=y_base_val,
        intervention_ids_to_coords=intervention_ids_to_coords,
        outdir=outdir,
        use_wandb=True
    )
    
    print("Displaying training curves...")
    plt.plot(clip_trainer.errors)
    plt.savefig(f'{outdir}/train.png')
    plt.clf()
    plt.plot(clip_trainer.validation_scores)
    plt.savefig(f'{outdir}/valid.png')

    # evaluate bias on held-out Concadia set
    print("Evaluating bias...")
    test_datasetIIT = get_IIT_concadia_dataset(
        split="test",
        tokenizer_name='openai/clip-vit-base-patch32',
        shuffle=False
    )
    X_base_test = test_datasetIIT[0]
    
    base_preds_test = clip_trainer.predict(
        X_base_test
    )
    
    base_preds_test = base_preds_test.view((-1, 2))
    y_preds_test = base_preds_test.argmax(dim=-1).detach().cpu().numpy()
    bias = y_preds_test.mean()
    
    print("Evaluating on flickr8k...")
    flickr8k_eval = evaluate_on_flickr8k(lim_clip)
    flickr8k_eval.to_csv(f'{outdir}/flickr8k.csv') # save right away
    print("Evaluating on food101...")
    food101_eval = evaluate_on_food101(lim_clip)
    print("Evaluating on imagenet...")
    imagenet_eval = evaluate_on_imagenet(lim_clip)
    print("Evaluating on sst2...")
    sst2_eval = evaluate_on_sst2(lim_clip)
    print("Evaluating on country211...")
    country211_eval = evaluate_on_country211(lim_clip)
    print("Evaluating on cifar100...")
    cifar100_eval = evaluate_on_cifar100(lim_clip)
    print("Evaluating cosid correlations...")
    cosid_correlations, length_correlation = evaluate_on_cosid(lim_clip)
    print("Evaluating integrated gradients...")
    # NOTE: set analysis mode to true before running IG
    lim_clip.set_analysis_mode(True)
    # NOT SURE IF THIS WORKS/MAKES ANY DIFFERENCE, BUT UNFREEZE ALL PARAMETERS?
    for param in lim_clip.parameters():
        param.requires_grad = True
    ig_correlations = evaluate_on_ig(lim_clip)

    # outdir = f'experiments/iit_finetune_lr{args.lr}_l2{args.l2}_epochs{args.max_iter}_seed{args.seed}'

    transfer_results = pd.DataFrame({
        'Food101': [food101_eval],
        'ImageNet': [imagenet_eval],
        'SST2': [sst2_eval],
        'Country211': [country211_eval],
        'CIFAR100': [cifar100_eval],
        'bias': [bias]
    })
    transfer_results.to_csv(f'{outdir}/transfer.csv')

    cosid_correlations.to_csv(f'{outdir}/cosid.csv')
    ig_correlations.to_csv(f'{outdir}/integrated_gradients.csv')

    length_correlation = pd.DataFrame({
        'R': [length_correlation[0]],
        'P value': [length_correlation[1]]
    })
    length_correlation.to_csv(f'{outdir}/length.csv')

    if args.save_model:
        outdir_model = f'{outdir}/model_weights.pt'
        torch.save(clip_trainer.model.state_dict(), outdir_model)

    wandb.finish()
    
        
if __name__ == '__main__':
    cmd = argparse.ArgumentParser()
    cmd.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    cmd.add_argument('--max_iter', default=20, type=int, help='number of epochs')
    cmd.add_argument('--l2', default=0., type=float, help='L2 regularization')
    cmd.add_argument('--early_stopping', default=False, action='store_true', help='early stopping tolerance')
    cmd.add_argument('--tol', default=1e-5, type=float, help='early stopping tolerance')
    cmd.add_argument('--n_iter_no_change', default=3, type=int, help='stop early if no improvement after this many iterations')
    cmd.add_argument('--num_iter_per_val', default=2000, type=int, help='how often to perform validation')
    cmd.add_argument('--train_size', default=None, type=int, help='Size of dataset (if None, uses the entire dataset)')
    cmd.add_argument('--val_size', default=1000, type=int, help='Size of dataset')
    cmd.add_argument('--lora', default=False, action='store_true', help='Run LoRA finetuning')
    cmd.add_argument('--rank', default=8, type=int, help='Rank for LoRA finetuning')
    cmd.add_argument('--lora_dropout', default=0., type=float, help='Dropout for LoRA finetuning')
    cmd.add_argument('--all_attention', default=False, action='store_true', help='Run LoRA on all attention layers')
    cmd.add_argument('--all_layers', default=False, action='store_true', help='Run LoRA finetuning on all linear layers')
    cmd.add_argument('--save_model', default=False, action='store_true', help='Check to save final model')
    cmd.add_argument('--seed', type=int, default=42, help='Set seed for reproducibility')
    cmd.add_argument('--out_dir', type=str, default='finetune_runs', help='Name of output directory')

    args = cmd.parse_args(sys.argv[1:])
    
    argument_strs = [f'{k}={v}' for k, v in vars(args).items()]
    print('Finetuning CLIP with', reduce(lambda s, t: s + ' ' + t, argument_strs))
    
    main(args)
