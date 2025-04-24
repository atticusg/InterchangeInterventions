import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import pearsonr
from transformers import AutoTokenizer, CLIPModel
from LIM_clip import LIMClipTextModel

COSID_IMAGE_FILE = '/nlp/scr/amirzur/cosid-image-encoding.pt'
COSID_DF_FILE = '/nlp/scr/amirzur/cosid_data/cosid-clipscores.csv'
BLV_RATINGS_FILE = '/nlp/scr/amirzur/cosid_data/blv_data_criticaltrials.csv'
SIGHTED_RATINGS_FILE = '/nlp/scr/amirzur/cosid_data/sighted_data_criticaltrials.csv'
VAR = 0
GROUP_TO_POSTFIX = {
    'BLV': '',
    'Sighted (no image)': '.preimg',
    'Sighted (with image)': '.postimg'
}

ASPECT_TO_COLUMN = {
    'Overall': 'q_overall',
    'Imaginability': 'q_reconstructivity',
    'Relevance': 'q_relevance',
    'Irrelevance': 'q_irrelevance'
}

class CosidDataset:
    def __init__(self, embed_func):
        self.embed_func = embed_func
        self.cosid_df = pd.read_csv(COSID_DF_FILE)

    def create_dataset(self):
        data = []
        for base in self.cosid_df['texts'].values:
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


def get_cosid_dataset(
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

    dataset = CosidDataset(
        embed_func=encoding
    )

    X_base, y_base = dataset.create_dataset()
    y_base = torch.tensor(y_base)
    return X_base, y_base

def get_human_data():
    df_blv_ratings = pd.read_csv(BLV_RATINGS_FILE)
    df_sighted_ratings = pd.read_csv(SIGHTED_RATINGS_FILE)
    df_sighted_ratings = df_sighted_ratings[df_sighted_ratings['description'].isin(df_blv_ratings['description'])]

    df_blv_avg = df_blv_ratings.melt(
        value_vars=['q_irrelevance', 'q_reconstructivity', 'q_relevance', 'q_imgfit', 'q_overall'],
        var_name='question',
        value_name='response',
        id_vars=['context', 'participant', 'img_id', 'description', 'article_text']
    ).groupby(
        ['description', 'img_id', 'context', 'question']
    ).mean().reset_index().rename(columns={'response': 'mean_response'})

    df_sighted_avg = df_sighted_ratings.melt(
        var_name='question',
        value_name='response',
        value_vars=[
            'q_relevance.preimg', 
            'q_irrelevance.preimg', 
            'q_reconstructivity.preimg', 
            'q_overall.preimg', 
            'q_relevance.postimg', 
            'q_imgfit.preimg', 
            'q_imgfit.postimg', 
            'q_irrelevance.postimg', 
            'q_overall.postimg'
        ],
        id_vars=['context', 'img_id', 'description', 'article_text']
    ).groupby(
        ['description', 'img_id', 'context', 'question']
    ).mean().reset_index().rename(columns={'response': 'mean_response'})

    return pd.concat((df_blv_avg, df_sighted_avg))

def compute_correlations(cosid_df):
    cosid_df = cosid_df[
        (cosid_df['imgstem_version'] != 'guitar') & (cosid_df['imgstem_version'] != '640px-parc_agen')
    ]
    print(cosid_df.columns)
    cosid_df[['imgstem', 'version']] = cosid_df['imgstem_version'].str.split('_version', 1, expand=True)
    cosid_df = cosid_df.drop(
        columns=['text_controls', 'imgstem_version']
    ).rename(columns={'texts': 'description'})

    df_human_data = get_human_data()

    df_clipscore_human_corr = df_human_data[[
        'description', 'img_id', 'context', 'question', 'mean_response'
    ]].merge(cosid_df, on='description')

    correlations = []
    for g, p in GROUP_TO_POSTFIX.items():
        for a, c in ASPECT_TO_COLUMN.items():
            # skip missing value for sighted imaginability
            if g == 'Sighted (with image)' and a == 'Imaginability':
                continue
            column = c + p
            df_group = df_clipscore_human_corr[df_clipscore_human_corr['question'] == column]
            r, p_val = pearsonr(df_group['mean_response'], df_group['prediction'])
            correlations.append({
                'Group': g,
                'Aspect': a,
                'PearsonR': r,
                'P-value': p_val
            })

    correlations = pd.DataFrame(correlations)
    length_correlation = pearsonr(
        df_clipscore_human_corr['description'].apply(len), 
        df_clipscore_human_corr['prediction']
    )

    return correlations, length_correlation

def evaluate_on_cosid(lim_clip):
    device = lim_clip.device
    X_base, y_base = get_cosid_dataset('openai/clip-vit-base-patch32')
    input, mask = X_base
    input = torch.stack(input).to(device).squeeze()
    mask = torch.stack(mask).to(device).squeeze()

    with torch.no_grad():
        text_embeds = lim_clip.get_text_encodings((input, mask))
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    image_embeds = torch.load(COSID_IMAGE_FILE).to(device)
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

    assert text_embeds.shape[0] == image_embeds.shape[0]
    clipscores = torch.matmul(text_embeds, image_embeds.squeeze(1).t()).diag().detach().cpu().numpy()

    cosid_df = pd.read_csv(COSID_DF_FILE)
    cosid_df['prediction'] = clipscores

    correlations_df, length_correlation = compute_correlations(cosid_df)

    return correlations_df, length_correlation


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

    print('Evaluating on Cosid...')
    correlations_df, length_correlation = evaluate_on_cosid(lim_clip)
    correlations_df.to_csv('cosid_correlations.csv')
    print(f'Length correlation: {length_correlation[0]:.2f} (p={length_correlation[1]:.2f})')


if __name__ == '__main__':
    main()
