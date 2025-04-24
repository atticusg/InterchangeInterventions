from datasets import load_dataset


print('Loading food101 dataset...')
# If the dataset is gated/private, make sure you have run huggingface-cli login
food101 = load_dataset("food101", split='validation', cache_dir='/nlp/scr/amirzur/.cache')

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print('Processing images...')
processed = processor(images=[im['image'] for im in food101], return_tensors='pt')

print('Embedding images...')

from tqdm import trange

batch_size = 8

outputs = None
with torch.no_grad():
    for b in trange(0, processed['pixel_values'].size(0), batch_size):
        batch = processed['pixel_values'][b: b + batch_size].to(device)
        image_features = model.get_image_features(pixel_values=batch)
        if outputs is None:
            outputs = image_features
        else:
            outputs = torch.cat((outputs, image_features))


print('Writing to file...')
torch.save(outputs, '/nlp/scr/amirzur/food101-encodings.pt')
