# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script can be used to search the best hyper parameters for training.
"""

import os
import json
import logging
import statistics
import wandb
from argparse import ArgumentParser
from collections import defaultdict

from train import train_pet
from data_utils import load_metrics
from cli import parser, process_args
from utils import set_seed
from model import TransformerModelWrapper
from config import load_pet_configs
from data_utils import TRAIN_SET, DEV_SET, DEV32_SET, TEST_SET, load_examples, load_metrics

logger = logging.getLogger('my_sweep')
os.environ['WANDB_API_KEY'] = 'be340acd2d3396078ae82c556d462b42a457dfad'


def get_best_results(metric, output_dir, result_file='results.json'):
    best_score, best_result, best_dir = -1.0, {}, ''
    for iter_dir in os.listdir(output_dir):
        full_name = os.path.join(output_dir, iter_dir, result_file)
        if os.path.exists(full_name):
            result = json.load(open(full_name, 'r'))['eval_set']
            if result[metric] > best_score:
                best_score, best_result, best_dir = result[metric], result, iter_dir

    return best_result, os.path.join(output_dir, best_dir)


def main():
    # Initialize wandb

    run = wandb.init(reinit=True, sync_tensorboard=True)

    config = wandb.config
    task, seed, encoder = config['task'], config['seed_split'], config['encoder']
    lr, wd, bs = config['learning_rate'], config['weight_decay'], config['batch_size']
    repeat, load_manual, extra_mask_rate = config['repeat'], config['load_manual'], config['extra_mask_rate']
    x_input, output_dir_suffix = config['x_input'], config['output_dir_suffix']
    warmup_lr, warmup, aug, soft_label = config['warmup_lr'], config['warmup'], config['aug'], config['soft_label']


    assert repeat > 0
    assert 0.0 <= extra_mask_rate < 0.5

    basic_arguments = ['--model_type', 'roberta',
                       '--embed_size', '1024',
                       '--do_train', '--do_eval',
                       '--eval_set', 'test',
                       '--overwrite_output_dir',
                       '--extra_mask_rate', str(extra_mask_rate)]


    logger.info('=== Task: %s ===' % task)
    best_result_all = defaultdict(list)
    best_result_stage1 = defaultdict(list)
    seed_list = [seed]
    for seed in seed_list:
        data_split = '16-%d' % seed
        if task == 'MNLI-mm':
            data_dir = os.path.join('/apdcephfs/private_shaotiancai/datasets/data', 'k-shot', 'MNLI', data_split)
        elif task == 'RTE-glue':
            data_dir = os.path.join('/apdcephfs/private_shaotiancai/datasets/data', 'k-shot', 'RTE', data_split)
        else:
            data_dir = os.path.join('/apdcephfs/private_shaotiancai/datasets/data', 'k-shot', task, data_split)
        # Change output directory name here!
        task_dir = os.path.join('/apdcephfs/private_shaotiancai/code/model/DART_copy/output', task, encoder)
        if x_input == 'replace':
            task_dir = os.path.join(task_dir, 'x_input-' + x_input, 'warmup-' + str(warmup))
        if x_input == 'mix':
            task_dir = os.path.join(task_dir, 'x_input-' + x_input, 'aug-' + str(aug))
        else:
            task_dir = os.path.join(task_dir, 'origin')

        if soft_label:
            task_dir = os.path.join(task_dir, 'soft')

        if output_dir_suffix:
            task_dir += '-' + output_dir_suffix

        print(task_dir)
        output_dir = os.path.join(task_dir, data_split)
        arguments = ['--task_name', task,
                     '--data_dir', data_dir,
                     '--pet_per_gpu_eval_batch_size', '256',
                     '--pet_max_steps', '1000',
                     '--pet_repetitions', str(repeat)]
        arguments.extend(['--learning_rate', str(lr), '--weight_decay', str(wd)])

        # Whether load pre-trained weights from manual prompt
        if load_manual:
            manual_output_dir = os.path.join(
                '/apdcephfs/private_shaotiancai/code/model/DART_copy/output', task, 'manual', data_split)
            _, best_dir = get_best_results(
                load_metrics(task.lower())[-1], manual_output_dir)
            arguments.extend(['--model_name_or_path', best_dir])
            logger.info("Load trained weights from %s..." % best_dir)
            output_dir = os.path.join(
                '/apdcephfs/private_shaotiancai/code/model/DART_copy/output', task, encoder, 'manual', data_split)
        else:
            arguments.extend(['--model_name_or_path', '/apdcephfs/private_shaotiancai/code/model/roberta-large',
                              '--cache_dir', '/apdcephfs/private_shaotiancai/code/model/roberta-large'])
        arguments.extend(['--output_dir', output_dir])

        if x_input:
            if task in ['MNLI', 'MNLI-mm', 'SNLI', 'RTE-glue']:
                arguments.extend(['--pet_max_seq_length', '256',
                                  '--pet_per_gpu_train_batch_size', str(bs),
                                  '--pet_gradient_accumulation_steps', '2'])
            else:
                arguments.extend(['--pet_max_seq_length', '128',
                                  '--pet_per_gpu_train_batch_size', str(bs),
                                  '--pet_gradient_accumulation_steps', '1'])
        else:
            if task in ['MNLI', 'MNLI-mm', 'SNLI', 'RTE-glue']:
                arguments.extend(['--pet_max_seq_length', '256',
                                  '--pet_per_gpu_train_batch_size', str(bs),
                                  '--pet_gradient_accumulation_steps', '2'])
            else:
                arguments.extend(['--pet_max_seq_length', '128',
                                  '--pet_per_gpu_train_batch_size', str(bs),
                                  '--pet_gradient_accumulation_steps', '1'])

        # Set prompt encoder type
        if encoder == 'inner2':
            arguments.extend(
                ['--prompt_encoder_type', 'inner', '--two_stage_train'])
        elif encoder == 'manual':
            arguments.extend(['--prompt_encoder_type', 'none'])
        else:
            arguments.extend(['--prompt_encoder_type', encoder])

        if x_input:
            arguments.extend(
                ['--x_input', x_input])

        if warmup:
            arguments.extend(
                ['--warmup'])

        if aug:
            arguments.extend(
                ['--aug'])

        if soft_label:
            arguments.extend(
                ['--soft_label'])

        if warmup_lr:
            arguments.extend(
                ['--warmup_lr', str(warmup_lr)])

        arguments.extend(
            ['--seed', str(seed)])

        args = parser.parse_args(basic_arguments + arguments)
        process_args(args)
        logger.info(args)

        if False and os.path.exists(os.path.join(output_dir, 'results.txt')):
            logger.info("Path %s already exists, skipping it..." %
                        output_dir)
        else:
            logger.info('--- Running data split: %s ---' % data_split)
            train_pet(args)

        # Load best result for current data split
        best_result, _ = get_best_results(args.metrics[-1], output_dir)
        for metric, value in best_result.items():
            best_result_all[metric].append(value)
        if args.two_stage_train:
            best_result, _ = get_best_results(
                args.metrics[-1], output_dir, 'results_stage1.json')
            for metric, value in best_result.items():
                best_result_stage1[metric].append(value)

        # Summary results
        logger.info("\n\n========== RESULTS OF TASK: %s ==========" % task)
        if args.two_stage_train:
            logger.info("---------- STAGE[1] RESULTS ----------")
            for metric, values in best_result_stage1.items():
                mean = statistics.mean(values)
                std = statistics.stdev(values) if len(values) > 1 else 0
                logger.info("{}: {:.1f}({:.1f})\n".format(
                    metric, mean * 100, std * 100))
            logger.info("---------- STAGE[2] RESULTS ----------")
        for metric, values in best_result_all.items():
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0
            logger.info("{}: {:.1f}({:.1f})\n".format(
                metric, mean * 100, std * 100))

    run.finish()


if __name__ == '__main__':
    main()
    # run_parser = ArgumentParser()
    # run_parser.add_argument("--task",
    #                         type=str, default='all',
    #                         choices=['SST-2', 'sst-5', 'mr', 'cr', 'mpqa', 'subj', 'trec', 'CoLA',
    #                                  'MNLI', 'MNLI-mm', 'SNLI', 'QNLI', 'RTE-glue', 'MRPC', 'QQP'])
    # run_parser.add_argument("--encoder",
    #                         type=str,
    #                         default='inner',
    #                         choices=['none', 'mlp', 'lstm', 'inner', 'inner2'])
    # run_parser.add_argument("--num_splits", type=int, default=-1)
    # run_parser.add_argument("--repeat", type=int, default=1)
    # run_parser.add_argument("--load_manual", action='store_true')
    # run_parser.add_argument("--extra_mask_rate", type=float, default=0.0)
    # run_parser.add_argument("--output_dir_suffix", "-o", type=str, default='')
    # run_parser.add_argument("--x_input",
    #                         choices=['replace', 'mix', 'mul'],
    #                         )
    # run_parser.add_argument("--warmup_lr", type=float, default=1e-4)
    # run_parser.add_argument("--warmup", action='store_true')
    # run_parser.add_argument("--aug", action='store_true')
    # run_parser.add_argument("--soft_label", action='store_true')
    # run_parser.add_argument("--seed_split",
    #                         type=int,
    #                         default=[],
    #                         nargs='+',
    #                         choices=[13, 21, 42, 87, 100])
    # run_parser.add_argument("--batch_size",
    #                         type=int,
    #                         default=[],
    #                         nargs='+',
    #                         choices=[4, 8, 16, 24, 32])
    # run_parser.add_argument("--sweep_id",
    #                         type=str,
    #                         default='')
    #
    # run_args = run_parser.parse_args()
    #
    # if not run_args.seed_split:  # Default search all seed splits
    #     run_args.seed_split = [13, 21, 42, 87, 100]
    #
    # if not run_args.batch_size:  # Default search all batch sizes
    #     if run_args.task in ['MNLI', 'MNLI-mm', 'SNLI', 'RTE-glue']:
    #         # Restrict maximum batch size due to memory limit
    #         run_args.batch_size = [4, 8, 16]
    #     else:
    #         run_args.batch_size = [4, 8, 16, 24, 32]
    #
    # # Prepare sweep config and get sweep id
    # sweep_config = {
    #     'program': run_args.task,
    #     'method': 'grid',
    #     'metric': {
    #         'goal': 'maximize',
    #         'name': 'eval-' + load_metrics(run_args.task)[-1]
    #     },
    #     'parameters': {
    #         'task': {'value': run_args.task},
    #         'encoder': {'value': run_args.encoder},
    #         'num_splits': {'value': run_args.num_splits},
    #         'repeat': {'value': run_args.repeat},
    #         'load_manual': {'value': run_args.load_manual},
    #         'extra_mask_rate': {'value': run_args.extra_mask_rate},
    #         'output_dir_suffix': {'value': run_args.output_dir_suffix},
    #         'x_input': {'value': run_args.x_input},
    #         'warmup_lr': {'value': run_args.warmup_lr},
    #         'warmup': {'value': run_args.warmup},
    #         'aug': {'value': run_args.aug},
    #         'soft_label': {'value': run_args.soft_label},
    #         'seed_split': {'values': run_args.seed_split},
    #         'learning_rate': {'values': [1e-6, 1e-5, 5e-5, 1e-4, 2e-4]},
    #         'weight_decay': {'values': [0.0, 0.01, 0.05, 0.10]},
    #         'batch_size': {'values': run_args.batch_size}
    #     }
    # }
    #
    # if run_args.sweep_id:  # Recover from old sweep
    #     sweep_id = run_args.sweep_id
    # else:  # Create new sweep
    #     sweep_id = wandb.sweep(sweep_config, project="SST-2", entity="szu_csse_bdi")
    #
    # # Sweep all hyper parameters
    # wandb.agent(sweep_id, function=main, project="SST-2", entity="szu_csse_bdi")
