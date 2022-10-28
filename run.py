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
from argparse import ArgumentParser
from collections import defaultdict

from data_utils import load_metrics
from cli import parser, process_args
from train import train_pet

logger = logging.getLogger('run')


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
    run_parser = ArgumentParser()
    run_parser.add_argument("--encoder",
                            choices=['manual', 'lstm', 'inner', 'inner2', 'mlp', 'sa'],
                            default='manual')
    run_parser.add_argument("--task", default='all')
    run_parser.add_argument("--num_splits", type=int, default=-1)
    run_parser.add_argument("--repeat", type=int, default=1)
    run_parser.add_argument("--load_manual", action='store_true')
    run_parser.add_argument("--extra_mask_rate", type=float, default=0.1)
    run_parser.add_argument("--output_dir_suffix", "-o", type=str, default='')
    run_parser.add_argument("--x_input",
                            choices=['replace', 'mix', 'mul'],
                            )
    run_parser.add_argument("--warmup_lr", type=float, default=1e-4)
    run_parser.add_argument("--warmup", action='store_true')
    run_parser.add_argument("--aug", action='store_true')
    run_parser.add_argument("--soft_label", action='store_true')

    run_parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam.")
    run_parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight decay if we apply some.")
    run_parser.add_argument("--pet_per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for PET training.")
    run_parser.add_argument("--mix_coef", default=1.0, type=float, help="mix calibration cof")
    run_parser.add_argument("--div_coef", default=1.0, type=float, help="diversity calibration cof")
    run_parser.add_argument("--k_shot", default=16, type=int, help="k-shot")
    run_parser.add_argument("--prompt_amp", type=int, help="prompt amplify count")
    run_parser.add_argument("--t5_spt", action='store_true', default=False, help="use T5 Sp tuning ?")
    run_parser.add_argument("--auto_pos", action='store_true', default=False, help="use auto position ?")
    run_parser.add_argument("--full_shot", action='store_true', default=False, help="use full shot ?")
    #run_parser.add_argument("--distill", action='store_true', default=False, help="use distill ?")

    run_args = run_parser.parse_args()

    seed_list = [13, 21, 42, 87, 100]

    single_tasks = ['SST-2', 'sst-5', 'mr',
                    'cr', 'mpqa', 'subj', 'trec', 'CoLA']
    pair_tasks = ['MNLI', 'MNLI-mm', 'SNLI',
                  'QNLI', 'rte-glue', 'MRPC', 'QQP']  # TODO: STS-B

    if run_args.task in single_tasks + pair_tasks:
        tasks = [run_args.task]
    elif run_args.task == 'single':
        tasks = single_tasks
    elif run_args.task == 'pair':
        tasks = pair_tasks
    elif run_args.task == 'all':
        tasks = single_tasks + pair_tasks
    else:
        raise NotImplementedError

    if run_args.num_splits >= 0:
        seed_list = [seed_list[run_args.num_splits]]
    elif run_args.num_splits != -1:
        raise NotImplementedError

    assert run_args.repeat > 0
    assert 0.0 <= run_args.extra_mask_rate < 0.5

    basic_arguments = ['--model_type', 'roberta',
                       '--embed_size', '1024',
                       '--do_train', '--do_eval',
                       '--eval_set', 'test',
                       '--overwrite_output_dir',
                       '--extra_mask_rate', str(run_args.extra_mask_rate)]

    for task in tasks:
        logger.info('=== Task: %s ===' % task)
        best_result_all = defaultdict(list)
        best_result_stage1 = defaultdict(list)
        for seed in seed_list:
            if not run_args.full_shot:
                data_split = '{}-{}'.format(run_args.k_shot, seed)
                if task == 'MNLI-mm':
                    data_dir = os.path.join('./data', 'k-shot', 'MNLI', data_split)
                elif task == 'rte-glue':
                    data_dir = os.path.join('./data', 'k-shot', 'RTE', data_split)
                else:
                    data_dir = os.path.join('./data', 'k-shot', task, data_split)
            else:
                data_dir = os.path.join('./data', 'original', task)
                data_split = 'full-shot'
            # Change output directory name here!
            task_dir = os.path.join('./output', task, run_args.encoder)
            if run_args.x_input == 'replace':
                task_dir = os.path.join(task_dir, 'x_input-' + run_args.x_input, 'warmup-' + str(run_args.warmup))
            if run_args.x_input == 'mix':
                task_dir = os.path.join(task_dir, 'x_input-' + run_args.x_input, 'aug-' + str(run_args.aug))
            else:
                task_dir = os.path.join(task_dir, 'origin')

            if run_args.soft_label:
                task_dir = os.path.join(task_dir, 'soft')

            if run_args.output_dir_suffix:
                task_dir += '-' + run_args.output_dir_suffix

            print(task_dir)
            output_dir = os.path.join(task_dir, data_split)
            arguments = ['--task_name', task,
                         '--data_dir', data_dir,
                         '--pet_per_gpu_eval_batch_size', '32',
                         '--pet_max_steps', '1000',   # 1000
                         '--pet_repetitions', str(run_args.repeat)]

            # Whether load pre-trained weights from manual prompt
            if run_args.load_manual:
                manual_output_dir = os.path.join(
                    './output', task, 'manual', data_split)
                _, best_dir = get_best_results(
                    load_metrics(task.lower())[-1], manual_output_dir)
                arguments.extend(['--model_name_or_path', best_dir])
                logger.info("Load trained weights from %s..." % best_dir)
                output_dir = os.path.join(
                    './output', task, run_args.encoder, 'manual', data_split)
            else:
                arguments.extend(['--model_name_or_path', '../model/roberta-large',
                                  '--cache_dir', './model/roberta-large'])
            arguments.extend(['--output_dir', output_dir])

            if run_args.x_input:
                if task in ['MNLI', 'MNLI-mm', 'SNLI', 'rte-glue']:
                    arguments.extend(['--pet_max_seq_length', '256',
                                      '--pet_per_gpu_train_batch_size', str(run_args.pet_per_gpu_train_batch_size),
                                      '--pet_gradient_accumulation_steps', '2'])
                else:
                    arguments.extend(['--pet_max_seq_length', '128',
                                      '--pet_per_gpu_train_batch_size', str(run_args.pet_per_gpu_train_batch_size),
                                      '--pet_gradient_accumulation_steps', '1'])
            else:
                if task in ['MNLI', 'MNLI-mm', 'SNLI', 'rte-glue']:
                    arguments.extend(['--pet_max_seq_length', '256',
                                      '--pet_per_gpu_train_batch_size', str(run_args.pet_per_gpu_train_batch_size),
                                      '--pet_gradient_accumulation_steps', '2'])
                else:
                    arguments.extend(['--pet_max_seq_length', '128',
                                      '--pet_per_gpu_train_batch_size', str(run_args.pet_per_gpu_train_batch_size),
                                      '--pet_gradient_accumulation_steps', '1'])

            if run_args.learning_rate:
                arguments.extend(['--learning_rate', str(run_args.learning_rate)])
            if run_args.weight_decay:
                arguments.extend(['--weight_decay', str(run_args.weight_decay)])

            # Set prompt encoder type
            if run_args.encoder == 'inner2':
                arguments.extend(
                    ['--prompt_encoder_type', 'inner', '--two_stage_train'])
            elif run_args.encoder == 'manual':
                arguments.extend(['--prompt_encoder_type', 'none'])
            else:
                arguments.extend(['--prompt_encoder_type', run_args.encoder])

            if run_args.x_input:
                arguments.extend(
                    ['--x_input', run_args.x_input])
                if run_args.x_input == 'mix':
                    assert run_args.mix_coef > 0
                    arguments.extend(
                            ['--mix_coef', str(run_args.mix_coef)])

            if run_args.prompt_amp:
                arguments.extend(
                        ['--prompt_amp', str(run_args.prompt_amp)])
                if run_args.t5_spt:
                    arguments.extend(
                            ['--t5_spt'])

            if run_args.auto_pos:
                arguments.extend(
                    ['--auto_pos'])

            if run_args.warmup:
                arguments.extend(
                    ['--warmup'])

            if run_args.aug:
                arguments.extend(
                        ['--aug'])
                assert run_args.mix_coef > 0
                arguments.extend(
                        ['--div_coef', str(run_args.div_coef)])

            if run_args.soft_label:
                arguments.extend(
                    ['--soft_label'])

            #if run_args.distill:
            #    arguments.extend(
            #        ['--distill'])

            if run_args.warmup_lr:
                arguments.extend(
                    ['--warmup_lr', str(run_args.warmup_lr)])

            arguments.extend(
                 ['--seed', str(seed)])

            args = parser.parse_args(basic_arguments + arguments)
            process_args(args)
            logger.info(args)

            if os.path.exists(os.path.join(output_dir, 'results.txt')):
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


if __name__ == '__main__':
    main()

