export CUDA_VISIBLE_DEVICES=0

for task in SST-2 sst-5 mr cr mpqa subj trec CoLA MNLI MNLI-mm SNLI QNLI RTE-glue MRPC QQP
do
    python run.py   \
        --encoder inner \
        --task $task   \
        --extra_mask_rate 0.1
done
#    --x_input   \
#    --warmup
