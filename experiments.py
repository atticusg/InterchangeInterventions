import os
import pandas as pd

# runs = ['runs/' + fn for fn in os.listdir('runs')]

# runs_das = [r for r in runs if 'das=True' and 'layer=10' in r]

# print(len(runs_das))

# # runs_lora = [r for r in runs_finetune if 'lora=True' in r]
# # runs_no_lora = [r for r in runs_finetune if 'lora=False' in r]

# best = 0
# best_results = None
# best_config = None
# for r in runs_das:
#     results = pd.read_csv(r + '/transfer.csv')
#     if results.iloc[0]['bias'] > best:
#         best = results.iloc[0]['bias']
#         best_results = results
#         best_config = r

# print(best_results)
# print(best_config)

def load_experiment(experiment_name, run_folder):
    df = None
    for folder in os.listdir(run_folder):
        if 'all_attention' not in folder:
            continue
        # TEMPORARY
        if ('finetune' in run_folder) and ('num_iter_per_val=250' not in folder):
            continue

        path = f'./{run_folder}/{folder}/{experiment_name}.csv'
        exp_df = pd.read_csv(path, index_col=0)

        # extract finetuning type
        if run_folder == 'finetune_seeds':
            exp_df['Finetuning Objective'] = 'Behavioral'
        elif run_folder == 'iit_lora_seeds':
            exp_df['Finetuning Objective'] = 'IIT'
        elif run_folder == 'das_lora_seeds':
            exp_df['Finetuning Objective'] = 'DAS'
        
        # extract LoRA parameters
        r_term = 'rank='
        r_ind = folder.find(r_term)
        r_end = folder.find('_', r_ind)
        exp_df['Rank'] = int(folder[r_ind + len(r_term): r_end]) # LoRA rank

        if 'all_layers=True' in folder:
            exp_df['Finetuning Method'] = 'LoRA (W_k, W_q, W_v, W_o, MLP)'
        elif 'all_attention=True' in folder:
            exp_df['Finetuning Method'] = 'LoRA (W_k, W_q, W_v, W_o)'
        elif 'lora=True' in folder:
            exp_df['Finetuning Method'] = 'W_q, W_v'
        else:
            exp_df['Finetuning Method'] = 'Finetuning'
            exp_df['Rank'] = '-'

        exp_df['Run'] = folder
        
        if df is None:
            df = exp_df
        else:
            df = pd.concat((df, exp_df))
    return df

def load_transfer_steps(run_folder):
    df = None
    for folder in os.listdir(run_folder):
        if 'all_attention=True' not in folder:
            continue
        # TEMPORARY
        if 'n_iter_no_change=100' not in folder:
            continue
        if ('finetune' in run_folder) and ('num_iter_per_val=250' not in folder):
            continue

        # extract seed
        s_term = 'seed='
        s_ind = folder.find(s_term)
        s_end = folder.find('_', s_ind)
        if s_end == -1:
            s_end = len(folder)
        seed = int(folder[s_ind + len(s_term): s_end])
        if seed >= 5:
            continue

        for filename in os.listdir(f'{run_folder}/{folder}'):
            if 'step' not in filename:
                continue

            path = f'./{run_folder}/{folder}/{filename}'
            step_df = pd.read_csv(path, index_col=0)

            # extract finetuning type
            if run_folder == 'finetune_seeds':
                step_df['Finetuning Objective'] = 'Behavioral'
            elif run_folder == 'iit_lora_seeds':
                step_df['Finetuning Objective'] = 'IIT'
            elif run_folder == 'das_lora_seeds':
                step_df['Finetuning Objective'] = 'DAS'

            step_df['Seed'] = seed

            step_df['Step'] = int(
                filename[len('transfer_step'): filename.find('.')]
            )
        
            # extract LoRA parameters
            # r_term = 'rank='
            # r_ind = folder.find(r_term)
            # r_end = folder.find('_', r_ind)
            # exp_df['Rank'] = int(folder[r_ind + len(r_term): r_end]) # LoRA rank

            if 'all_layers=True' in folder:
                step_df['Finetuning Method'] = 'LoRA (W_k, W_q, W_v, W_o, MLP)'
            elif 'all_attention=True' in folder:
                step_df['Finetuning Method'] = 'LoRA (W_k, W_q, W_v, W_o)'
            elif 'lora=True' in folder:
                step_df['Finetuning Method'] = 'W_q, W_v'
            else:
                step_df['Finetuning Method'] = 'Finetuning'
                # exp_df['Rank'] = '-'

            step_df['Run'] = folder
            
            if df is None:
                df = step_df
            else:
                df = pd.concat((df, step_df))
    return df

# for experiment_name in ['integrated_gradients']: #['cosid', 'integrated_gradients', 'length', 'transfer']:
#     df = None
#     transfer_df = None
#     for run_folder in ['finetune_seeds', 'iit_lora_seeds', 'das_lora_seeds']:
#         exp_df = load_experiment(experiment_name, run_folder)
#         if df is None:
#             df = exp_df
#         else:
#             df = pd.concat((df, exp_df))

#         transfer_steps_df = load_transfer_steps(run_folder)
#         if transfer_df is None:
#             transfer_df = transfer_steps_df
#         else:
#             transfer_df = pd.concat((transfer_df, transfer_steps_df))
#     outdir = 'compiled_lora_seeds'
#     os.makedirs(outdir, exist_ok=True)
#     df.to_csv(f'{outdir}/{experiment_name}.csv', index=False)
#     transfer_df.to_csv(f'{outdir}/transfer_steps.csv')

outdir = 'compiled_lora_seeds'
transfer_df = load_transfer_steps('finetune_seeds')
transfer_df.to_csv(f'{outdir}/transfer_steps_long.csv')

exp_df = load_experiment('integrated_gradients', 'iit_lora_seeds')
exp_df.to_csv(f'{outdir}/integrated_gradients_iit.csv')