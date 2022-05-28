export CUDA_VISIBLE_DEVICES=0

python sweep.py   \
--encoder mlp \
--task SST-2   \
--x_input mix
