import os
import torch
import numpy as np
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, CLIPModel
from LIM_clip import LIMClipTextModel

from datasets import load_dataset

from tqdm import tqdm, trange

IMAGENET_FILE = '/nlp/scr/amirzur/imagenet1k-image-encoding.pt'
IMAGENET_CLASSES = '/nlp/scr/amirzur/imagenet1k-labels.txt'
IMAGENET_LABELS = '/nlp/scr/amirzur/imagenet1k-labeled.txt'
VAR = 0

class ImagenetDataset:
    def __init__(self, embed_func):
        self.embed_func = embed_func
        with open(IMAGENET_CLASSES) as f:
            self.class_labels = [
                line.strip().split(',')[0] for line in f
            ]

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


def get_imagenet_dataset(
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
    
    dataset = ImagenetDataset(
        embed_func=encoding
    )
    
    X_base, y_base = dataset.create_dataset()
    y_base = torch.tensor(y_base)
    return X_base, y_base


def evaluate_on_imagenet(lim_clip):
    device = lim_clip.device
    X_base, y_base = get_imagenet_dataset('openai/clip-vit-base-patch32')
    input, mask = X_base
    input = torch.stack(input).to(device).squeeze()
    mask = torch.stack(mask).to(device).squeeze()

    batch_size = 16
    text_embeds = None
    with torch.no_grad():
        for b in trange(0, input.shape[0], batch_size, desc="Imagenet text embeddings"):
            embeds = lim_clip.get_text_encodings((
                input[b: b + batch_size], mask[b: b + batch_size]
            ))
            if text_embeds is None:
                text_embeds = embeds
            else:
                text_embeds = torch.cat((text_embeds, embeds))
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    
    print('Loading Imagenet image embeddings...')
    image_embeds = torch.load(IMAGENET_FILE).to(device)
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    
    labels = np.fromfile(IMAGENET_LABELS, dtype=np.int32)
    
    preds = None
    for b in trange(0, image_embeds.shape[0], batch_size, desc="Imagenet logits"):
        logits_per_text = torch.matmul(text_embeds, image_embeds[b: b + batch_size].squeeze(1).t())
        logits_per_image = logits_per_text.t()
        pred_batch = logits_per_image.argmax(dim=-1).detach().cpu().numpy()
        if preds is None:
            preds = pred_batch
            print(logits_per_text.shape)
        else:
            preds = np.concatenate((preds, pred_batch))

    print(preds.shape, labels.shape)

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
    
    print('Evaluating on Imagenet...')
    print('Model F1:', evaluate_on_imagenet(lim_clip))


if __name__ == '__main__':
    main()
