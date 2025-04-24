import os
import torch
import numpy as np
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, CLIPModel
from LIM_clip import LIMClipTextModel

from datasets import load_dataset

FOOD_101_FILE = '/nlp/scr/amirzur/food101-image-encoding.pt'
FOOD_101_LABELS = '/nlp/scr/amirzur/food101-labels.txt'
VAR = 0

class Food101Dataset:
    def __init__(self, embed_func):
        self.embed_func = embed_func
        food101 = load_dataset('food101', split='validation', cache_dir='/nlp/scr/amirzur/.cache')
        self.class_labels = food101.features['label'].names

    def create_dataset(self):
        data = []
        for class_label in self.class_labels:
            base = f'A picture of {class_label}'
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


def get_food101_dataset(
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
    
    dataset = Food101Dataset(
        embed_func=encoding
    )
    
    X_base, y_base = dataset.create_dataset()
    y_base = torch.tensor(y_base)
    return X_base, y_base


def evaluate_on_food101(lim_clip):
    device = lim_clip.device
    X_base, y_base = get_food101_dataset('openai/clip-vit-base-patch32')
    input, mask = X_base
    input = torch.stack(input).to(device).squeeze()
    mask = torch.stack(mask).to(device).squeeze()

    with torch.no_grad():
        text_embeds = lim_clip.get_text_encodings((input, mask))
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    
    image_embeds = torch.load(FOOD_101_FILE).to(device)
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

    labels = np.fromfile(FOOD_101_LABELS, dtype=np.int64)
    
    logits_per_text = torch.matmul(text_embeds, image_embeds.squeeze(1).t())
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
    
    print('Evaluating on Food 101...')
    print('Model F1:', evaluate_on_food101(lim_clip))


if __name__ == '__main__':
    main()
