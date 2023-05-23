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
import ray
from ray import tune

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


def main(config):
    #run_parser = ArgumentParser()
    #run_parser.add_argument("--encoder",
    #                        choices=['manual', 'lstm', 'inner', 'inner2', 'mlp', 'sa'],
    #                        default='manual')
    #run_parser.add_argument("--task", default='all')
    #run_parser.add_argument("--num_splits", type=int, default=-1)
    #run_parser.add_argument("--repeat", type=int, default=1)
    #run_parser.add_argument("--load_manual", action='store_true')
    #run_parser.add_argument("--extra_mask_rate", type=float, default=0.1)
    #run_parser.add_argument("--output_dir_suffix", "-o", type=str, default='')
    #run_parser.add_argument("--x_input",
    #                        choices=['replace', 'mix', 'mul'],
    #                        )
    #run_parser.add_argument("--warmup_lr", type=float, default=1e-4)
    #run_parser.add_argument("--warmup", action='store_true')
    #run_parser.add_argument("--aug", action='store_true')
    #run_parser.add_argument("--soft_label", action='store_true')

    #run_parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam.")
    #run_parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight decay if we apply some.")
    #run_parser.add_argument("--pet_per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for PET training.")
    #run_parser.add_argument("--mix_coef", default=1.0, type=float, help="mix calibration cof")
    #run_parser.add_argument("--div_coef", default=1.0, type=float, help="diversity calibration cof")
    #run_parser.add_argument("--k_shot", default=16, type=int, help="k-shot")
    #run_parser.add_argument("--prompt_amp", type=int, help="prompt amplify count")
    #run_parser.add_argument("--t5_spt", action='store_true', default=False, help="use T5 Sp tuning ?")
    #run_parser.add_argument("--auto_pos", action='store_true', default=False, help="use auto position ?")
    #run_parser.add_argument("--full_shot", action='store_true', default=False, help="use full shot ?")
    ##run_parser.add_argument("--distill", action='store_true', default=False, help="use distill ?")

    #run_args = run_parser.parse_args()

    task, num_splits, repeat, extra_mask_rate = 'QNLI', 0, 1, 0.0 
    full_shot, k_shot, encoder, x_input, warmup = False, 16, 'sa', 'replace', True
    aug, soft_label, output_dir_suffix, load_manual, pet_per_gpu_train_batch_size = True, False, '', False, 8
    learning_rate, weight_decay, mix_coef, div_coef, prompt_amp, t5_spt, auto_pos = 1e-4, 0.05, 1.0, 1.0, None, False, False
    warmup_lr = 1e-4

    seed_list = [13, 21, 42, 87, 100]

    single_tasks = ['SST-2', 'sst-5', 'mr',
                    'cr', 'mpqa', 'subj', 'trec', 'CoLA']
    pair_tasks = ['MNLI', 'MNLI-mm', 'SNLI',
                  'QNLI', 'rte-glue', 'MRPC', 'QQP']  # TODO: STS-B

    if task in single_tasks + pair_tasks:
        tasks = [task]
    elif task == 'single':
        tasks = single_tasks
    elif task == 'pair':
        tasks = pair_tasks
    elif task == 'all':
        tasks = single_tasks + pair_tasks
    else:
        raise NotImplementedError

    if num_splits >= 0:
        if not config['num_splits']:
            seed_list = [seed_list[num_splits]]
        else:
            seed_list = [seed_list[config['num_splits']]]
    elif num_splits != -1:
        raise NotImplementedError

    assert repeat > 0
    assert 0.0 <= extra_mask_rate < 0.5

    basic_arguments = ['--model_type', 'roberta',
                       '--embed_size', '1024',
                       '--do_train', '--do_eval',
                       '--eval_set', 'test',
                       '--overwrite_output_dir',
                       '--extra_mask_rate', str(extra_mask_rate)]

    for task in tasks:
        logger.info('=== Task: %s ===' % task)
        best_result_all = defaultdict(list)
        best_result_stage1 = defaultdict(list)
        for seed in seed_list:
            if not full_shot:
                data_split = '{}-{}'.format(k_shot, seed)
                if task == 'MNLI-mm':
                    data_dir = os.path.join('/lxm/lt/SLSP-master-with-distill-amp/data', 'k-shot', 'MNLI', data_split)
                elif task == 'rte-glue':
                    data_dir = os.path.join('/lxm/lt/SLSP-master-with-distill-amp/data', 'k-shot', 'RTE', data_split)
                else:
                    data_dir = os.path.join('/lxm/lt/SLSP-master-with-distill-amp/data', 'k-shot', task, data_split)
            else:
                data_dir = os.path.join('/lxm/lt/SLSP-master-with-distill-amp/data', 'original', task)
                data_split = 'full-shot'
            # Change output directory name here!
            task_dir = os.path.join('./output', task, encoder)
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
                         '--pet_per_gpu_eval_batch_size', '32',
                         '--pet_max_steps', '1000',   # 1000
                         '--pet_repetitions', str(repeat)]

            # Whether load pre-trained weights from manual prompt
            if load_manual:
                manual_output_dir = os.path.join(
                    './output', task, 'manual', data_split)
                _, best_dir = get_best_results(
                    load_metrics(task.lower())[-1], manual_output_dir)
                arguments.extend(['--model_name_or_path', best_dir])
                logger.info("Load trained weights from %s..." % best_dir)
                output_dir = os.path.join(
                    './output', task, encoder, 'manual', data_split)
            else:
                arguments.extend(['--model_name_or_path', '/lxm/lt/model/roberta-large',
                                  '--cache_dir', '/lxm/lt/model/roberta-large'])
            arguments.extend(['--output_dir', output_dir])

            if x_input:
                if task in ['MNLI', 'MNLI-mm', 'SNLI', 'rte-glue']:
                    if not config['pet_per_gpu_train_batch_size']:
                        arguments.extend(['--pet_max_seq_length', '256',
                            '--pet_per_gpu_train_batch_size', str(pet_per_gpu_train_batch_size),
                            '--pet_gradient_accumulation_steps', '2'])
                    else:
                        arguments.extend(['--pet_max_seq_length', '256',
                            '--pet_per_gpu_train_batch_size', str(config['pet_per_gpu_train_batch_size']),
                            '--pet_gradient_accumulation_steps', '2'])
                else:
                    if not config['pet_per_gpu_train_batch_size']:
                        arguments.extend(['--pet_max_seq_length', '128',
                            '--pet_per_gpu_train_batch_size', str(pet_per_gpu_train_batch_size),
                            '--pet_gradient_accumulation_steps', '1'])
                    else:
                        arguments.extend(['--pet_max_seq_length', '128',
                            '--pet_per_gpu_train_batch_size', str(config['pet_per_gpu_train_batch_size']),
                            '--pet_gradient_accumulation_steps', '1'])
            else:
                if task in ['MNLI', 'MNLI-mm', 'SNLI', 'rte-glue']:
                    if not config['pet_per_gpu_train_batch_size']:
                        arguments.extend(['--pet_max_seq_length', '256',
                            '--pet_per_gpu_train_batch_size', str(pet_per_gpu_train_batch_size),
                            '--pet_gradient_accumulation_steps', '2'])
                    else:
                        arguments.extend(['--pet_max_seq_length', '256',
                            '--pet_per_gpu_train_batch_size', str(config['pet_per_gpu_train_batch_size']),
                            '--pet_gradient_accumulation_steps', '2'])
                else:
                    if not config['pet_per_gpu_train_batch_size']:
                        arguments.extend(['--pet_max_seq_length', '128',
                            '--pet_per_gpu_train_batch_size', str(pet_per_gpu_train_batch_size),
                            '--pet_gradient_accumulation_steps', '1'])
                    else:
                        arguments.extend(['--pet_max_seq_length', '128',
                            '--pet_per_gpu_train_batch_size', str(config['pet_per_gpu_train_batch_size']),
                            '--pet_gradient_accumulation_steps', '1'])

            if learning_rate:
                if not config['learning_rate']:
                    arguments.extend(['--learning_rate', str(learning_rate)])
                else:
                    arguments.extend(['--learning_rate', str(config['learning_rate'])])

            if weight_decay:
                if not config['weight_decay']:
                    arguments.extend(['--weight_decay', str(weight_decay)])
                else:
                    arguments.extend(['--weight_decay', str(config['weight_decay'])])

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
                if x_input == 'mix':
                    assert mix_coef > 0
                    arguments.extend(
                            ['--mix_coef', str(mix_coef)])

            if prompt_amp:
                arguments.extend(
                        ['--prompt_amp', str(prompt_amp)])
                if t5_spt:
                    arguments.extend(
                            ['--t5_spt'])

            if auto_pos:
                arguments.extend(
                    ['--auto_pos'])

            if warmup:
                arguments.extend(
                    ['--warmup'])

            if aug:
                arguments.extend(
                        ['--aug'])
                assert mix_coef > 0
                arguments.extend(
                        ['--div_coef', str(div_coef)])

            if soft_label:
                arguments.extend(
                    ['--soft_label'])

            #if distill:
            #    arguments.extend(
            #        ['--distill'])

            if warmup_lr:
                arguments.extend(
                    ['--warmup_lr', str(warmup_lr)])

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

def sub_main(config):
    main(config)

if __name__ == '__main__':
    config = {
            #"learning_rate": tune.sample_from(lambda _: np.random.randint(2, 5)),
            "learning_rate": tune.choice([1e-6, 1e-5, 5e-5, 1e-4, 2e-4]),
            "pet_per_gpu_train_batch_size": tune.choice([4, 8, 16, 24, 32]),
            "num_splits": tune.choice([0, 1, 2, 3, 4]),
            "weight_decay": tune.choice([0.0, 0.1, 0.01, 0.05]),
            }
    config_t = {
            #"learning_rate": tune.sample_from(lambda _: np.random.randint(2, 5)),
            "learning_rate": 1,
            "pet_per_gpu_train_batch_size": 1,
            "num_splits": 1,
            "weight_decay": 1,
            }
    #main(config_t)
    ray.init()
    result = tune.run(
            sub_main,
            resources_per_trial={"gpu": 1, 'cpu':8},
            config=config,
            #num_samples=20,
            )
    print("======================== Result =========================")
    print(result.results_df)

