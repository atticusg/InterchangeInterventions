for seed in 0 1 2 3 4
do
    python finetune_clip.py --lr 5e-06 --l2 0.0 --max_iter 10 --num_iter_per_val 500 --val_size 5000 --out_dir "seeds_bft" --seed $seed
done