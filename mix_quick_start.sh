python -m pdb run.py   \
    --encoder lstm \
    --task SST-2 \
    --pet_per_gpu_train_batch_size 8 \
    --learning_rate 1e-5    \
    --weight_decay 0.05  \
    --num_splits 0 \
    --x_input mix   \
    --mix_coef 1 \
    --div_coef 10   \
    --prompt_amp 3
#    --soft_label    \
#    --aug   \
