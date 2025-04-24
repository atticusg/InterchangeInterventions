import os
import torch
import numpy as np
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, CLIPModel
from LIM_clip import LIMClipTextModel

NEG_IMAGE_FILE = '/nlp/scr/amirzur/rendered-ss2-negative-image-encoding.pt'
POS_IMAGE_FILE = '/nlp/scr/amirzur/rendered-ss2-positive-image-encoding.pt'
VAR = 0

class SST2Dataset:
    def __init__(self, embed_func):
        self.embed_func = embed_func
        self.class_labels = ['negative', 'positive']

    def create_dataset(self):
        data = []
        for class_label in self.class_labels:
            base = f'A picture of text with {class_label} sentiment'
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


def get_sst2_dataset(
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
    
    dataset = SST2Dataset(
        embed_func=encoding
    )
    
    X_base, y_base = dataset.create_dataset()
    y_base = torch.tensor(y_base)
    return X_base, y_base


def evaluate_on_sst2(lim_clip):
    device = lim_clip.device
    X_base, y_base = get_sst2_dataset('openai/clip-vit-base-patch32')
    input, mask = X_base
    input = torch.stack(input).to(device).squeeze()
    mask = torch.stack(mask).to(device).squeeze()

    with torch.no_grad():
        text_embeds = lim_clip.get_text_encodings((input, mask))
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    pos_embeds = torch.load(POS_IMAGE_FILE)
    neg_embeds = torch.load(NEG_IMAGE_FILE)

    image_embeds = torch.cat((pos_embeds, neg_embeds))
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

    labels = np.concatenate((np.ones(pos_embeds.shape[0]), np.zeros(neg_embeds.shape[0])))
    
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
    
    print('Evaluating on SST2...')
    print('Model F1:', evaluate_on_sst2(lim_clip))


if __name__ == '__main__':
    main()
