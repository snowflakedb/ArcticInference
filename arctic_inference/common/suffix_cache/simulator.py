# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import itertools
import multiprocessing as mp
import os
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from arctic_inference.common.suffix_cache import SuffixCache

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def suffix_decode(
    suffix_cache: SuffixCache,
    request_id: int,
    prompt: List[int],
    ground_truth_response: List[int],
    max_spec_tokens: int,
    max_spec_factor: float,
    min_token_prob: float,
    use_tree_spec: bool,
    use_cached_prompt: bool,
) -> List[Dict]:
    if not max_spec_tokens:
        max_spec_tokens = suffix_cache.max_depth

    if use_cached_prompt:
        suffix_cache.cache_prompt(request_id, prompt)

    assert isinstance(prompt, list) and isinstance(ground_truth_response, list)

    results = []
    response = []
    while len(response) < len(ground_truth_response):
        text = prompt + response

        start_time = time.perf_counter()
        result = suffix_cache.speculate(
            request_id,
            text,
            max_spec_tokens=max_spec_tokens,
            max_spec_factor=max_spec_factor,
            min_token_prob=min_token_prob,
            use_tree_spec=use_tree_spec,
            use_cached_prompt=use_cached_prompt,
        )
        end_time = time.perf_counter()
        spec_time = end_time - start_time

        # Verify scpeculated tokens
        accepted_tokens = []
        node = -1
        for token_id in ground_truth_response[len(response):]:
            children = [i for i, p in enumerate(result.parents) if p == node]
            for c in children:
                if result.token_ids[c] == token_id:
                    accepted_tokens.append(token_id)
                    node = c
                    break
            else:
                break

        new_tokens = accepted_tokens.copy()
        response.extend(accepted_tokens)
        assert len(response) <= len(ground_truth_response)
        if len(response)  < len(ground_truth_response):
            # Add bonus token
            bonus_token = ground_truth_response[len(response)]
            new_tokens.append(bonus_token)
            response.append(bonus_token)

        # Update suffix cache
        start_time = time.perf_counter()
        suffix_cache.update_response(request_id, new_tokens)
        end_time = time.perf_counter()
        update_time = end_time - start_time

        results.append({
            "step": len(results),
            "match_len": result.match_len,
            "score": result.score,
            "num_spec_toks": len(result.token_ids),
            "num_accept_toks": len(accepted_tokens),
            "num_out_toks": len(new_tokens),
            "spec_ms": spec_time * 1000,
            "update_ms": update_time * 1000,
        })

    assert response == ground_truth_response

    if use_cached_prompt:
        suffix_cache.evict_prompt(request_id)

    return results


def sample_data(
    dataset: pd.DataFrame,
    train_dataset: Optional[pd.DataFrame],
    num_eval: Optional[int],
    num_train: Optional[int],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if train_dataset is None:
        if num_eval is None:
            num_eval = len(dataset) - num_train
        if num_train is None:
            num_train = len(dataset) - num_eval
        assert num_train + num_eval <= len(dataset)
        shuffled = dataset.sample(frac=1, random_state=seed)
        train_subset = shuffled.head(num_train)
        eval_subset = shuffled.tail(num_eval)
    else:
        if num_eval is None:
            num_eval = len(dataset)
            eval_subset = dataset
        else:
            shuffled = dataset.sample(frac=1, random_state=seed)
            eval_subset = shuffled.head(num_eval)
        if num_train is None:
            num_train = len(train_dataset)
            train_subset = train_dataset
        else:
            shuffled = train_dataset.sample(frac=1, random_state=seed)
            train_subset = shuffled.head(num_train)
    return eval_subset, train_subset


def process_task(
    dataset: pd.DataFrame,
    train_dataset: Optional[pd.DataFrame],
    task_id: int,
    num_eval: int,
    num_train: int,
    seed: int,
    max_depth: int,
    max_spec_tokens: int,
    max_spec_factor: float,
    min_token_prob: float,
    use_tree_spec: bool,
    use_cached_prompt: bool,
) -> List[Dict]:
    eval_subset, train_subset = sample_data(
        dataset,
        train_dataset,
        num_eval,
        num_train,
        seed,
    )
    suffix_cache = SuffixCache(max_depth)
    for request_id, example in tqdm(train_subset.iterrows(),
                                    total=len(train_subset),
                                    desc=f"Building cache"):
        # Use negative request_id to indicate training examples and avoid
        # conflicts with eval request_ids numbered 0, .., num_eval - 1.
        suffix_cache.update_response(-1 - request_id + 1, example["response"])

    records = []
    for request_id, example in tqdm(eval_subset.iterrows(),
                                    total=len(eval_subset),
                                    desc=f"Running requests"):
        results = suffix_decode(
            suffix_cache,
            request_id,
            example["prompt"],
            example["response"],
            max_depth,
            max_spec_factor=max_spec_factor,
            min_token_prob=min_token_prob,
            use_tree_spec=use_tree_spec,
            use_cached_prompt=use_cached_prompt,
        )
        for result in results:
            result.update({
                "task_id": task_id,
                "request_id": request_id,
                "num_eval": len(eval_subset),
                "num_train": len(train_subset),
                "seed": seed,
                "max_depth": max_depth,
                "max_spec_tokens": max_spec_tokens,
                "max_spec_factor": max_spec_factor,
                "min_token_prob": min_token_prob,
                "use_tree_spec": use_tree_spec,
                "use_cached_prompt": use_cached_prompt,
            })
        records.extend(results)

    return records


def results_summary(df: pd.DataFrame, config_cols: List[str]) -> pd.DataFrame:
    # Compute per-request speedup.
    speedup = df.groupby(["task_id", "request_id"]).agg(
        sum_out_toks=("num_out_toks", "sum"),
        num_steps=("step", "count"),
    )
    speedup["speedup"] = speedup["sum_out_toks"] / speedup["num_steps"]
    speedup = speedup.groupby(["task_id"]).agg(
        req_speedup=("speedup", "mean"),
    )
    # Compute summary statistics.
    config_cols = ["task_id"] + list(config_cols)
    summary = df.groupby(config_cols).agg(
        sum_accept_toks=("num_accept_toks", "sum"),
        sum_spec_toks=("num_spec_toks", "sum"),
        sum_out_toks=("num_out_toks", "sum"),
        avg_accept_toks=("num_accept_toks", "mean"),
        avg_spec_toks=("num_spec_toks", "mean"),
        sum_spec_ms=("spec_ms", "sum"),
        sum_update_ms=("update_ms", "sum"),
    ).reset_index()
    summary["accept_rate"] = (
        summary["sum_accept_toks"] / summary["sum_spec_toks"])
    summary["req_speedup"] = speedup["req_speedup"]
    summary["spec_ms_per_tok"] = (
        summary["sum_spec_ms"] / summary["sum_spec_toks"])
    summary["update_ms_per_tok"] = (
        summary["sum_update_ms"] / summary["sum_out_toks"])
    # Calculate columns to drop from the summary
    drop_cols = [col for col in config_cols if summary[col].nunique() == 1]
    drop_cols.extend([
        "sum_accept_toks",
        "sum_spec_toks",
        "sum_out_toks",
        "sum_spec_ms",
        "sum_update_ms"])
    return summary.drop(columns=drop_cols).set_index("task_id")


def read_data_file(
    path: str,
    prompt_column: str,
    response_column: str,
    format: Optional[str] = None,
) -> pd.DataFrame:
    # Read the dataset file into a pandas DataFrame.
    if format is None:
        _, ext = os.path.splitext(path)
        format = ext[1:]
    if format == "csv":
        df = pd.read_csv(path)
    elif format == "json":
        df = pd.read_json(path)
    elif format == "jsonl":
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported dataset format: {format}")
    # Ensure that the prompt and response columns are present in the dataset.
    if prompt_column not in df.columns:
        raise ValueError(f"Column '{prompt_column}' not found in dataset")
    if response_column not in df.columns:
        raise ValueError(f"Column '{response_column}' not found in dataset")
    # Drop all columns except the prompt and response columns.
    df = df[[prompt_column, response_column]]
    return df.rename(columns={
        prompt_column: "prompt",
        response_column: "response",
    })


def tokenize_data(dataset: pd.DataFrame, tokenizer_name: str) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    prompts = []
    responses = []
    for _, row in tqdm(dataset.iterrows(), total=len(dataset),
                       desc="Tokenizing dataset"):
        prompts.append(tokenizer.encode(row["prompt"]))
        responses.append(tokenizer.encode(row["response"]))
    return pd.DataFrame({
        "prompt": prompts,
        "response": responses,
    })


def ensure_tokenized(dataset: pd.DataFrame):
    for _, row in dataset.iterrows():
        if not isinstance(row["prompt"], list):
            break
        if not all(isinstance(x, int) for x in row["prompt"]):
            break
        if not isinstance(row["response"], list):
            break
        if not all(isinstance(x, int) for x in row["response"]):
            break
    else:
        return
    raise ValueError(
        "Dataset must be tokenized or a tokenizer must be provided")


def get_data(args: argparse.Namespace) -> Tuple[pd.DataFrame,
                                                Optional[pd.DataFrame]]:
    dataset = read_data_file(args.dataset, args.prompt_column,
                             args.response_column, args.format)
    max_num_eval = max(args.num_eval) if args.num_eval else 1
    max_num_train = max(args.num_train) if args.num_train else 0
    if args.train_dataset is not None:
        train_dataset = read_data_file(args.train_dataset, args.prompt_column,
                                       args.response_column, args.format)
        if args.num_eval and max_num_eval > len(dataset):
            raise ValueError(
                f"Number of evaluation examples ({max_num_eval}) exceeds the "
                f"size of the dataset ({len(dataset)})"
            )
        if args.num_train and max_num_train > len(train_dataset):
            raise ValueError(
                f"Number of training examples ({max_num_train}) exceeds the "
                f"size of the training dataset ({len(train_dataset)})"
            )
    else:
        train_dataset = None
        if max_num_eval + max_num_train > len(dataset):
            raise ValueError(
                f"Number of evaluation examples ({max_num_eval}) and training "
                f"examples ({max_num_train}) exceed the size of the dataset "
                f"({len(dataset)})"
            )
    return dataset, train_dataset


def main(args: argparse.Namespace):
    dataset, train_dataset = get_data(args)
    # Tokenize datasets (if needed)
    if args.tokenizer is not None:
        dataset = tokenize_data(dataset, args.tokenizer)
        if train_dataset is not None:
            train_dataset = tokenize_data(train_dataset, args.tokenizer)
    else:
        ensure_tokenized(dataset)
        if train_dataset is not None:
            ensure_tokenized(train_dataset)
    # Create all possible configurations
    num_eval = args.num_eval or [None]
    num_train = args.num_train or [None]
    configs = OrderedDict(
        num_eval=num_eval,
        num_train=num_train,
        seed=args.seed,
        max_depth=args.max_depth,
        max_spec_tokens=args.max_spec_tokens,
        max_spec_factor=args.max_spec_factor,
        min_token_prob=args.min_token_prob,
        use_tree_spec=args.use_tree_spec,
        use_cached_prompt=args.use_cached_prompt,
    )
    config_values = itertools.product(*configs.values())
    config_values = [
        (dataset, train_dataset, i, *v) for i, v in enumerate(config_values)]

    records = []
    with mp.Pool(args.parallel) as pool:
        for results in pool.starmap(process_task, config_values):
            records.extend(results)

    print("Preparing results...")

    df = pd.DataFrame.from_records(records)

    summary = results_summary(df, list(configs.keys()))
    print("\nSummary of results:\n")
    print(summary.to_string() + "\n")

    if args.output is not None:
        df.to_csv(args.output, index=False)
        print(f"Detailed results saved to: {args.output}")


def bool_arg(v):
    if v.lower() not in ("true", "false"):
        raise ValueError(f"Invalid boolean argument '{v}'")
    return v.lower() == "true"


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=str,
        help="Path to the dataset file",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "jsonl", "csv"],
        help="Format of the dataset file, uses its extension if not provided",
    )
    parser.add_argument(
        "--train-dataset",
        type=str,
        help="Path to a separate dataset file for training",
    )
    parser.add_argument(
        "--prompt-column",
        type=str,
        default="prompt",
        help="Column name for the prompts in the dataset",
    )
    parser.add_argument(
        "--response-column",
        type=str,
        default="response",
        help="Column name for the responses in the dataset",
    )
    parser.add_argument(
        "--num-train",
        type=int,
        nargs="+",
        help=("Number of examples to use for training (required if "
              "separate --train-dataset is not provided)"),
    )
    parser.add_argument(
        "--num-eval",
        type=int,
        nargs="+",
        help=("Number of examples to use for evaluation (required if "
              "separate --train-dataset is not provided)"),
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=[0],
        help="Random seed (for train/eval split)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name of the HuggingFace tokenizer",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="The path to the output CSV file",
    )
    parser.add_argument(
        "-p",
        "--parallel",
        type=int,
        default=16,
        help="Max number of parallel processes",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        nargs="+",
        default=[64],
        help="Max depth of the suffix tree",
    )
    parser.add_argument(
        "--max-spec-tokens",
        type=int,
        nargs="+",
        default=[0],
        help="Max speculation tokens (if 0, defaults to max_depth)",
    )
    parser.add_argument(
        "--max-spec-factor",
        type=float,
        nargs="+",
        default=[1.0],
        help="Max speculation tokens as a multiplier of the prefix length",
    )
    parser.add_argument(
        "--min-token-prob",
        type=float,
        nargs="+",
        default=[0.1],
        help="Minimum probability of the token to be considered",
    )
    parser.add_argument(
        "--use-tree-spec",
        type=bool_arg,
        nargs="*",
        default=[True],
        help="Whether to use tree-based speculation (True/False)",
    )
    parser.add_argument(
        "--use-cached-prompt",
        type=bool_arg,
        nargs="*",
        default=[True],
        help=("Whether to use the cached prompt for the request in addition "
              "to the global cache of previous responses (True/False)"),
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if args.train_dataset is None:
        if args.num_train is None and args.num_eval is None:
            raise ValueError("Must provide --num-train or --num-eval if "
                             "separate --train-dataset is not provided")
    if len(args.use_tree_spec) == 0:
        args.use_tree_spec = [True]
    if len(args.use_cached_prompt) == 0:
        args.use_cached_prompt = [True]
    main(args)
