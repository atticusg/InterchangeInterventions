import os, sys, json, random, argparse, itertools
from functools import reduce

import pandas as pd
import numpy as np
import torch
from torchvision.datasets import CIFAR100

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from transformers import AutoTokenizer, CLIPModel

from LIM_clip import LIMClipTextModel
# from trainer import CLIPLIMTrainer
from utils import fix_random_seeds

SEED = 42
VAR = 0

class CIFAR100Dataset:
    def __init__(self, cifar100, embed_func):
        self.embed_func = embed_func
        self.cifar100 = cifar100

    def create_dataset(self):
        data = []
        for class_label in self.cifar100.classes:
            base = f'a photo of a {class_label}'
            base_x, base_mask = self.embed_func(base)
            base_label = 0   # base label ignored during training
            data.append((base_x, base_mask, base_label))

        base, base_mask, y = zip(*data)
        self.base = base
        self.base_mask = base_mask
        self.y = np.array(y)
        return (
            (self.base, self.base_mask), 
            self.y
        )

def get_cifar_100_dataset(
        cifar100,
        tokenizer_name
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

        dataset = CIFAR100Dataset(
            cifar100,
            embed_func=encoding,
        )

        X_base, y_base = dataset.create_dataset()
        y_base = torch.tensor(y_base)
        return X_base, y_base

def evaluate_on_cifar100(lim_clip):
    cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
    X_base, y_base = get_cifar_100_dataset(cifar100, 'openai/clip-vit-base-patch32')
    input, mask = X_base
    input = torch.stack(input).to(lim_clip.device).squeeze()
    mask = torch.stack(mask).to(lim_clip.device).squeeze()

    with torch.no_grad():
        text_embeds = lim_clip.get_text_encodings((input, mask))

    image_embeds = torch.load('cifar100-image-encoding.pt')
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    logits_per_text = torch.matmul(text_embeds, image_embeds.squeeze(1).t())
    logits_per_image = logits_per_text.t()
    labels = [l for _, l in cifar100]
    preds = logits_per_image.argmax(dim=-1).detach().cpu().numpy()
    return f1_score(y_true=labels, y_pred=preds, average='macro')


def main():
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
    )

    print('Evaluating on CIFAR 100...')
    print('Model F1:', evaluate_on_cifar100(lim_clip))


if __name__ == '__main__':
    main()
