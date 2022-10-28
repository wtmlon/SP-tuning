python -m pdb run.py   \
    --encoder lstm \
    --task QNLI \
    --pet_per_gpu_train_batch_size 4 \
    --learning_rate 1e-5    \
    --extra_mask_rate 0.1 \
    --weight_decay 0.1  \
    --x_input replace  \
    --warmup    \
    --warmup_lr 1e-4  \
    --mix_coef 1 \
    --div_coef 10   \
    --aug
#    --num_splits 4 \
#    --prompt_amp 3 
#    --soft_label    \
#    --auto_pos  \
#    --t5_spt    \
