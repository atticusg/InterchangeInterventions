for seed in 0 1 2 3 4
do
    python run_iit_clip.py --das --l2 0 --lr 1e-05 --layer 10 --intervention_size 256 --max_iter 10 --train_size 100000 --val_size 5000 --num_iter_per_val 500 --out_dir "seeds_das" --lora --all_attention --rank 64 --seed $seed
done