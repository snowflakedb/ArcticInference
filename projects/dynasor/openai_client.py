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

"""Dynasor OpenAI client API example"""

import argparse
import logging
from openai import OpenAI


def init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = init_logger()


def parse_args():
    parser = argparse.ArgumentParser(description="OpenAI Chat Client")
    parser.add_argument("--api-key", default="EMPTY", help="OpenAI API key")
    parser.add_argument(
        "--base-url", default="http://localhost:8000/v1", help="OpenAI API base URL"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048, help="Maximum tokens for completion"
    )
    parser.add_argument(
        "--disable-dynasor",
        action="store_true",
        help="Disable Dynasor"
    )
    parser.add_argument(
        "--probe-interval",
        type=int,
        default=32,
        help="Probe interval for adaptive compute"
    )
    parser.add_argument(
        "--certainty-window",
        type=int,
        default=2,
        help="Certainty window for adaptive compute"
    )
    parser.add_argument("--prompt", default="2+2=", help="User prompt")
    # parser.add_argument("--no-stream", action="store_true", help="Do not stream the response")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.debug("Args: %s", args)

    # stream = not args.no_stream
    # assert stream, "No streaming implementation is not supported yet."
    stream = True
    disable_dynasor = args.disable_dynasor

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        max_retries=1,
    )

    logger.debug("Grab models")
    models = client.models.list()
    model = models.data[0].id
    logger.debug("Model: %s", model)

    dynasor_config = dict()
    if not disable_dynasor:
        dynasor_config = dict(
            probe_interval=args.probe_interval,
            certainty_window=args.certainty_window,
        )

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": args.prompt},
        ],
        model=model,
        max_tokens=args.max_tokens,
        extra_body={
            "dynasor": dynasor_config
        },
        stream=stream,
    )

    print("Prompt: ", args.prompt)
    print("-" * 10)
    print("Response: \n")
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)


if __name__ == "__main__":
    main()
