import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, CLIPModel
from LIM_clip import LIMClipTextModel
from tqdm import tqdm, trange

COUNTRIES_DIR = '/nlp/scr/amirzur/'
VAR = 0

class Country211Dataset:
    def __init__(self, embed_func):
        self.embed_func = embed_func
        self.iso_df = pd.read_csv(COUNTRIES_DIR + '/all.csv')
        countries = sorted([fn for fn in os.listdir(COUNTRIES_DIR) if 'country211-' in fn])
        # convert country iso codes to country names
        country_names = []
        for country in countries:
            iso = country[country.find('-') + 1:country.find('-') + 3]
            row = self.iso_df[self.iso_df['alpha-2'] == iso]
            # unfortunate case where Namibia (NA) is interpreted as null n/a
            if row.shape[0] == 0:
                country_names.append('Namibia')
            else:
                country_names.append(row.iloc[0]['name'])
        # class labels are in sorted alphabetical order by country code
        self.class_labels = country_names

    def create_dataset(self):
        data = []
        for class_label in self.class_labels:
            base = f'A picture of the country of {class_label}'
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


def get_country211_dataset(
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
    
    dataset = Country211Dataset(
        embed_func=encoding
    )
    
    X_base, y_base = dataset.create_dataset()
    y_base = torch.tensor(y_base)
    return X_base, y_base


def evaluate_on_country211(lim_clip):
    device = lim_clip.device
    X_base, y_base = get_country211_dataset('openai/clip-vit-base-patch32')
    input, mask = X_base
    input = torch.stack(input).to(device).squeeze()
    mask = torch.stack(mask).to(device).squeeze()

    with torch.no_grad():
        text_embeds = lim_clip.get_text_encodings((input, mask))
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    country_embeds_files = sorted([fn for fn in os.listdir(COUNTRIES_DIR) if 'country211-' in fn])
    country_embeds_list = [
        torch.load(COUNTRIES_DIR + '/' + country_file)
        for country_file in country_embeds_files
    ]
    country_embeds = torch.cat(country_embeds_list)

    labels = np.concatenate([
        np.full(country_embeds_list[i].shape[0], i)
        for i in range(len(country_embeds_list))
    ])
    
    logits_per_text = None
    batch_size = 512
    for b in trange(0, len(country_embeds), batch_size, desc="Country211 logits"):
        image_embeds = country_embeds[b: b + batch_size]
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        if logits_per_text is None:
            logits_per_text = torch.matmul(text_embeds, image_embeds.squeeze(1).t())
            print(logits_per_text.shape)
        else:
            logits_per_text = torch.concat((
                logits_per_text, 
                torch.matmul(text_embeds, image_embeds.squeeze(1).t())
            ), dim=-1)

    print(f'{text_embeds.shape} x {country_embeds.shape} = {logits_per_text.shape}')

    logits_per_image = logits_per_text.t()
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
    
    print('Evaluating on Country211...')
    print('Model F1:', evaluate_on_country211(lim_clip))


if __name__ == '__main__':
    main()
