for acc in 1
do
    for seed in 3
	do
        for lr in 1e-4
		do
			for bz in 32 24 16 8 4
			do
				for wd in 0.0 0.1 0.05 0.01 0.001
				do
					python -m pdb run.py   \
						--encoder sa \
						--task QNLI\
						--x_input replace  \
						--learning_rate $lr    \
						--pet_per_gpu_train_batch_size $bz \
						--weight_decay $wd  \
						--num_splits $seed \
						--pet_gradient_accumulation_steps $acc  \
                        --aug
				done
			done
		done
	done
done
#    --prompt_amp 3 
#    --soft_label    \
#    --auto_pos  \
#    --t5_spt    \
#    --mix_coef 1 \
#    --div_coef 10   \
#    --num_splits 3 \
#                --warmup    \
#                --warmup_lr 1e-4  \
#                --extra_mask_rate 0.1 \
#for lr in 1e-3 2e-3 1e-7 2e-7 1e-8
