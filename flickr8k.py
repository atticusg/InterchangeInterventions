import os
import json
from tqdm import trange
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import kendalltau
from transformers import AutoTokenizer, CLIPModel
from LIM_clip import LIMClipTextModel

FLICKR8K_IMAGE_FILE = '/nlp/u/amirzur/InterchangeInterventions/flickr8k-image-encoding.pt'
FLICKR8K_DATA_FILE = '/nlp/u/amirzur/InterchangeInterventions/flickr8k.json'

class Flickr8kDataset:
    def __init__(self, embed_func):
        self.embed_func = embed_func

    def create_dataset(self):
        with open(FLICKR8K_DATA_FILE) as f:
            dataset = json.load(f)
        data = []
        for k, v in list(dataset.items()):
            for human_judgement in v['human_judgement']:
                if np.isnan(human_judgement['rating']):
                    print('NaN')
                    continue
                base_x, base_mask = self.embed_func(human_judgement['caption'])
                base_label = human_judgement['rating']
                data.append((base_x, base_mask, base_label))
        base, base_mask, y = zip(*data)
        self.base = base
        self.base_mask = base_mask
        self.y = np.array(y)
        return (
            (self.base, self.base_mask),
            self.y
        )


def get_flickr8k_dataset(
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

    dataset = Flickr8kDataset(
        embed_func=encoding
    )

    X_base, y_base = dataset.create_dataset()
    y_base = torch.tensor(y_base)
    return X_base, y_base

def evaluate_on_flickr8k(lim_clip, batch_size=16):
    device = lim_clip.device
    X_base, y_base = get_flickr8k_dataset('openai/clip-vit-base-patch32')
    input, mask = X_base
    input = torch.stack(input).to(device).squeeze()
    mask = torch.stack(mask).to(device).squeeze()

    text_embeds = None
    with torch.no_grad():
        for b in trange(0, input.shape[0], batch_size, desc="Flickr8k text embeddings"):
            embeds = lim_clip.get_text_encodings((
                input[b: b + batch_size], mask[b: b + batch_size]
            ))
            if text_embeds is None:
                text_embeds = embeds
            else:
                text_embeds = torch.cat((text_embeds, embeds))
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    image_embeds = torch.load(FLICKR8K_IMAGE_FILE).to(device)
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

    assert text_embeds.shape[0] == image_embeds.shape[0]
    clipscores = torch.matmul(text_embeds, image_embeds.squeeze(1).t()).diag().detach().cpu().numpy()

    correlation = kendalltau(clipscores, y_base, variant='c')

    return pd.DataFrame({
        'correlation': [correlation.statistic],
        'p-value': [correlation.pvalue]
    })


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    iit_layer = 10
    hidden_dim_per_concept = 128
    model_dim = 512

    intervention_ids_to_coords = {
        0: [{"layer":iit_layer, "start":0, "end":hidden_dim_per_concept}]
    }

    lim_clip = LIMClipTextModel(
        clip,
        max_length=77,
        device=device,
        target_layers=[iit_layer],
        target_dims={ "start": 0, "end": model_dim },
    )

    print('Evaluating on Flickr8k...')
    correlations_df = evaluate_on_flickr8k(lim_clip)
    correlations_df.to_csv('flickr8k_correlations.csv')
    print(f'Correlation: {correlations_df.correlation[0]:.2f}')


if __name__ == '__main__':
    main()
