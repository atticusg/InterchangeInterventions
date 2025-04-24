from datasets import load_dataset


print('Loading imagenet dataset...')
# If the dataset is gated/private, make sure you have run huggingface-cli login
imagenet = load_dataset("imagenet-1k", split='validation', cache_dir='/nlp/scr/amirzur/.cache')

print('Success!')
