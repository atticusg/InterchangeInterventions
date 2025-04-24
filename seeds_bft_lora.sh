for seed in 0 1 2 3 4
do
    python finetune_clip.py --lr 1e-05 --l2 0.0 --max_iter 10 --num_iter_per_val 500 --val_size 5000 --out_dir "seeds_bft" --lora --all_attention --rank 128 --seed $seed
done