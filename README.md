# SP-tuning
Implementation for paper * A Fine-grained Self-adapting Prompt Learning Approach for Few-shot Learning with Pre-trained Language Models.
## Environment
- python@3.6
- Use `pip install -r requirements.txt` to install dependencies.
- `wandb` account is required if the user wants to search for best hyper-parameter combinations.
## Data source
- 16-shot GLUE dataset from [LM-BFF](https://github.com/princeton-nlp/LM-BFF).
- Generated data consists of 5 random splits (13/21/42/87/100) for a task, each has 16 samples.
## How to run
- To run across each 5 splits in a task, use `run.py`:
    - `sh mix_quick_start.sh` for mix type prompt
    - `sh replace_quick_start.sh` for replace type prompt
```bash
$ python run.py -h
usage: run.py [-h] [--encoder {manual,lstm,inner,inner2,mlp}] [--task TASK] [--num_splits NUM_SPLITS] [--repeat REPEAT] [--load_manual] [--extra_mask_rate EXTRA_MASK_RATE]
                   [--output_dir_suffix OUTPUT_DIR_SUFFIX] [--x_input {replace,mix,mul}] [--warmup_lr WARMUP_LR] [--warmup] [--aug] [--soft_label] [--learning_rate LEARNING_RATE]
                   [--weight_decay WEIGHT_DECAY] [--pet_per_gpu_train_batch_size PET_PER_GPU_TRAIN_BATCH_SIZE]

optional arguments:
        -h, --help          
            show this help message and exit
        --encoder {manual,lstm,inner,inner2,mlp}
        --task TASK
        --num_splits NUM_SPLITS
        --repeat REPEAT
        --load_manual
        --extra_mask_rate EXTRA_MASK_RATE
        --output_dir_suffix OUTPUT_DIR_SUFFIX, -o OUTPUT_DIR_SUFFIX
        --x_input {replace,mix,mul}
        --warmup_lr WARMUP_LR
        --warmup
        --aug
        --soft_label
        --learning_rate LEARNING_RATE
            The initial learning rate for Adam.
        --weight_decay WEIGHT_DECAY
            Weight decay if we apply some.
        --pet_per_gpu_train_batch_size PET_PER_GPU_TRAIN_BATCH_SIZE
            Batch size per GPU/CPU for PET training.

```
- To train and evaluate on a single split with details recorded, use `inference.py`.
  - Before running, [`task_name`, `label_list`, `prompt_type`] should be configured in the code.
- To find optimal hyper-parameters for each task-split and reproduce our result, please use `sweep.py`:
  - Please refer to documentation for [WandB](https://docs.wandb.ai/) for more details.
```bash
$ python sweep.py -h
usage: sweep.py [-h] [--task {SST-2,sst-5,mr,cr,mpqa,subj,trec,CoLA,MNLI,MNLI-mm,SNLI,QNLI,RTE-glue,MRPC,QQP}] [--encoder {none,mlp,lstm,inner,inner2}]
                     [--seed_split {13,21,42,87,100} [{13,21,42,87,100} ...]] [--batch_size {4,8,16,24,32} [{4,8,16,24,32} ...]] [--sweep_id SWEEP_ID] [--x_input X_INPUT]

optional arguments:
        -h, --help            
            show this help message and exit
        --task {SST-2,sst-5,mr,cr,mpqa,subj,trec,CoLA,MNLI,MNLI-mm,SNLI,QNLI,RTE-glue,MRPC,QQP}
        --encoder {none,mlp,lstm,inner,inner2}
        --seed_split {13,21,42,87,100} [{13,21,42,87,100} ...]
        --batch_size {4,8,16,24,32} [{4,8,16,24,32} ...]
        --sweep_id SWEEP_ID
        --x_input X_INPUT
```
