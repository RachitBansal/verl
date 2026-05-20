# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
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
Preprocess the math dataset to parquet format
"""

import os
import datasets

import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/n/netscratch/sham_lab/Everyone/cmohri/data/rl_cbs')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'aime-1983-2024'

    dataset = datasets.load_dataset('gneubig/aime-1983-2024', trust_remote_code=True)

    test_dataset = dataset['train']
    # instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example['Question']

            solution = example.pop("Answer")
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "system",
                    "content": "Please reason step by step, and put your final answer within \\boxed{}."
                }, {
                    "role": "user",
                    "content": question
                }
                ],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn('train'), with_indices=True)

    local_dir = os.path.expanduser(args.local_dir)
    hdfs_dir = args.hdfs_dir

    os.makedirs(local_dir, exist_ok=True)
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    print(f"Data source: gneubig/aime-1983-2024")
    print(f"Length of test dataset: {len(test_dataset)}")

    if hdfs_dir is not None:
        from verl.utils.hdfs_io import copy, makedirs
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)