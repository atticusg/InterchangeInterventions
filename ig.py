import json
from tqdm import tqdm
import pandas as pd
import torch
from scipy.stats import pearsonr
from transformers import AutoTokenizer, CLIPModel
from LIM_clip import LIMClipTextModel

from captum.attr import visualization as viz
from captum.attr import LayerIntegratedGradients
from integrated_gradients_patch import LayerIntegratedGradients as LayerIntegratedGradientsGuided

CONCADIA_PATH = 'concadia.pt' 
CONCADIA_METADATA_PATH = 'wiki_split.json'
CONCRETENESS_PATH = '/nlp/scr/amirzur/concreteness.csv'
IMAGEABILITY_PATH = '/nlp/scr/amirzur/imageability.csv'
COLUMNS = ['filename', 'caption', 'description', 'image_features']
VAR = 0
SEED = 42
N = 100

def hf_ig_encodings(text, tokenizer):
    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.bos_token_id
    sep_id = tokenizer.eos_token_id
    inputs = tokenizer(text, truncation=True, max_length=77)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    base_ids = [cls_id] + [pad_id] * (len(input_ids) - 2) + [sep_id]
    return torch.LongTensor([input_ids]), torch.LongTensor([base_ids]), torch.LongTensor([attention_mask])

def hf_ig_analysis_one(text, image_encoding, tokenizer, lim_clip, guided=False, is_mediated=True):
    layer = lim_clip.embeddings

    input_ids, base_ids, attention_mask = hf_ig_encodings(text, tokenizer)

    def ig_forward(inputs):
        return lim_clip((
            inputs, 
            attention_mask.expand(inputs.shape).to(lim_clip.device), 
            image_encoding.to(lim_clip.device)
        ), output_hidden_states=guided)
    
    if guided:
        ig = LayerIntegratedGradientsGuided(ig_forward, layer, is_mediated=is_mediated)
    else:
        ig = LayerIntegratedGradients(ig_forward, layer)

    attrs = ig.attribute(
        input_ids.to(lim_clip.device),
        base_ids.to(lim_clip.device),
        target=None,
        return_convergence_delta=False,
        attribute_to_layer_input=False
    )

    scores = attrs.sum(dim=-1).squeeze(0)
    
    # question: should we normalize attributes?
    # scores = (scores - scores.mean()) / scores.norm()

    raw_input = tokenizer.convert_ids_to_tokens(input_ids.tolist()[0])

    pred_prob = lim_clip((
        input_ids.to(lim_clip.device), 
        attention_mask.to(lim_clip.device), 
        image_encoding.to(lim_clip.device)
    )).item()

    score_vis = viz.VisualizationDataRecord(
        word_attributions=scores,
        pred_prob=pred_prob,
        pred_class=None,
        true_class=None,
        attr_class=None,
        # override attribution label with aspect label, to make the visualization clearer
        # attr_class=ASPECTS[aspect] if aspect is not None else None,
        attr_score=attrs.sum(),
        raw_input_ids=raw_input,
        convergence_score=None
    )

    return score_vis

def clean_token(t):
    extra = t.find('</w>')
    if extra == -1:
        return t
    return t[:extra]

def hf_ig_analysis(texts, image_encodings, tokenizer, lim_clip, guided=False, is_mediated=True):
    results = []
    for text, image_encoding in tqdm(zip(texts, image_encodings), total=len(texts)):
        result = hf_ig_analysis_one(text, image_encoding, tokenizer, lim_clip, guided=guided, is_mediated=is_mediated)
        tokens = [clean_token(t) for t in result.raw_input_ids[1:-1]]
        attributions = result.word_attributions[1:-1].detach().cpu().tolist()
        assert(len(tokens) == len(attributions))
        results += [[t, a] for t, a in zip(tokens, attributions)]
    return pd.DataFrame.from_records(results, columns=['text', 'attribution'])

def get_concadia_test_dataset(tokenizer):
    concadia_data = torch.load(CONCADIA_PATH, map_location='cpu')
    concadia_df = pd.DataFrame(concadia_data, columns=COLUMNS).set_index('filename')
    with open(CONCADIA_METADATA_PATH) as f:
        concadia_metadata = json.load(f)['images']
    test_filenames = [im_data['filename'] for im_data in concadia_metadata if im_data['split'] == 'test']
    test_df = concadia_df.loc[test_filenames]
    return test_df


def compute_correlation(attr_df, value_df):
    capt_attr_df = pd.merge(attr_df, value_df, left_on='text', right_on='Words')
    capt_attr_df = capt_attr_df.groupby('text').mean()

    assert ('IMAG' in capt_attr_df.columns) or ('Conc.M' in capt_attr_df.columns)
    if 'IMAG' in capt_attr_df.columns:
        return pearsonr(capt_attr_df['attribution'], capt_attr_df['IMAG'])
    else:
        return pearsonr(capt_attr_df['attribution'], capt_attr_df['Conc.M'])

def evaluate_on_ig(lim_clip):
    concreteness_df = pd.read_csv(CONCRETENESS_PATH)[['Word', 'Conc.M']].rename(columns={
        'Word': 'Words'
    })
    imageability_df = pd.read_csv(IMAGEABILITY_PATH)
    imageability_df = imageability_df[['Words', 'IMAG']].iloc[1:] # skip first line
    imageability_df['IMAG'] = imageability_df['IMAG'].astype(float)
    merged_df = pd.merge(left=imageability_df, right=concreteness_df, left_on='Words', right_on='Words')
    
    tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-base-patch32')
    test_df = get_concadia_test_dataset(tokenizer)
    relevant = test_df.apply(
        lambda r: any([w in merged_df['Words'].values for w in r.caption.split()]) and \
            any([w in merged_df['Words'].values for w in r.description.split()]),
        axis=1
    )
    relevant.sum()
    test_df = test_df[relevant].sample(N, random_state=SEED)

    caption_attribution = hf_ig_analysis(
        test_df.caption.tolist() + test_df.description.tolist(), 
        test_df.image_features.tolist() + test_df.image_features.tolist(),
        tokenizer, lim_clip,
        guided=False
    )

    assert lim_clip.analysis, 'Must put LIM CLIP in analysis mode for integrated gradients'
    
    caption_attribution_guided = hf_ig_analysis(
        test_df.caption.tolist() + test_df.description.tolist(), 
        test_df.image_features.tolist() + test_df.image_features.tolist(),
        tokenizer, lim_clip,
        guided=True
    )

    caption_attribution_unmediated = hf_ig_analysis(
        test_df.caption.tolist() + test_df.description.tolist(),
        test_df.image_features.tolist() + test_df.image_features.tolist(),
        tokenizer, lim_clip,
        guided=True, is_mediated=False
    )

    concreteness_unguided = compute_correlation(caption_attribution, concreteness_df)
    imageability_unguided = compute_correlation(caption_attribution, imageability_df)
    concreteness_guided = compute_correlation(caption_attribution_guided, concreteness_df)
    imageability_guided = compute_correlation(caption_attribution_guided, imageability_df)
    concreteness_unmediated = compute_correlation(caption_attribution_unmediated, concreteness_df)
    imageability_unmediated = compute_correlation(caption_attribution_unmediated, imageability_df)

    return pd.DataFrame([
        {
            'Mediation': 'Full',
            'Value': 'Concreteness',
            'R': concreteness_unguided[0],
            'P value': concreteness_unguided[1]
        },
        {
            'Mediation': 'Through',
            'Value': 'Concreteness',
            'R': concreteness_guided[0],
            'P value': concreteness_guided[1]
        },
        {
            'Mediation': 'Around',
            'Value': 'Concreteness',
            'R': concreteness_unmediated[0],
            'P value': concreteness_unmediated[1]
        },
        {
            'Mediation': 'Full',
            'Value': 'Imageability',
            'R': imageability_unguided[0],
            'P value': imageability_unguided[1]
        },
        {
            'Mediation': 'Through',
            'Value': 'Imageability',
            'R': concreteness_guided[0],
            'P value': concreteness_guided[1]
        },
        {
            'Mediation': 'Around',
            'Value': 'Imageability',
            'R': imageability_unmediated[0],
            'P value': imageability_unmediated[1]
        }
    ])


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
        target_dims={ "start": 0, "end": model_dim }
    )

    lim_clip.set_analysis_mode(True)

    eye_size = lim_clip.analysis_model.layers[iit_layer + 1].weight.shape[0]
    lim_clip.analysis_model.layers[iit_layer + 1].weight = torch.eye(eye_size).to(device)
    lim_clip.load_state_dict(torch.load('results/iit_das_lr5e-06_epochs1_layer10_size128_tsize30000.pt'))

    print('Evaluating on Integrated Gradients...')
    correlations_df = evaluate_on_ig(lim_clip)
    correlations_df.to_csv('ig_correlations.csv')

if __name__ == '__main__':
    main()
