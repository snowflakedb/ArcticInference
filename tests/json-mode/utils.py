"""Util functions to query Corvo service."""

import time

from config import (
    EMBEDDING_MODEL,
    REQUEST_MAX_RETRIES,
    REQUEST_TIMEOUT_IN_SECONDS,
    URL_CORVO_COMPLETE_XP,
    URL_CORVO_EMBED_XP,
)
from loguru import logger
import numpy as np
import requests


def call_corvo_embed(prompts: list[str]) -> list[np.ndarray] | list[None]:
    """Return the embeddings of each submitted prompt (or None if it failed)."""
    payload = {
        "data": [
            [i, EMBEDDING_MODEL, prompt] for i, prompt in enumerate(prompts)
        ]
    }
    try:
        response = requests.post(
            url=URL_CORVO_EMBED_XP,
            json=payload,
            headers={
                "sf-external-function-signature": "(MODEL VARCHAR, TEXT VARCHAR)",
                "sf-ml-account-hash": "internal-eval",
            },
            timeout=REQUEST_TIMEOUT_IN_SECONDS,
        )
        response.raise_for_status()
        data = response.json()["data"]
    except Exception as e:
        logger.error(
            f"Generating the embeddings failed. Use None as embeddings instead.({e})."
        )
        return [None for _ in range(len(prompts))]

    return [np.array(embedding) for _, embedding in data]


def call_corvo_complete(
    prompts: list[list[dict[str, str]]],
    llm_name: str,
    options: dict[str, float | dict],
) -> requests.Response:
    """Send the prompts (batch of prompts) to the LLM. Return the parsed response."""
    data = [[i, llm_name, prompt, options] for i, prompt in enumerate(prompts)]
    payload = {"data": data}
    headers = {
        "sf-ml-enabled-complete-guardrails": "false",
        "sf-external-function-signature": "(MODEL VARCHAR, MESSAGES ARRAY, OPTIONS OBJECT)",
        "sf-external-function-name": "TRY_COMPLETE$V2",
        "sf-ml-enabled-complete-json-mode": "true",
        "sf-ml-account-hash": "internal-eval",
        "sf-ml-enabled-cross-regions": "ANY_REGION",
    }

    retry_count = 0
    backoff_time = 10  # Start with 10 second backoff

    while retry_count <= REQUEST_MAX_RETRIES:
        try:
            llm_response = requests.post(
                url=URL_CORVO_COMPLETE_XP,
                json=payload,
                headers=headers,
                timeout=REQUEST_TIMEOUT_IN_SECONDS,
            )

            # Handle 422 errors - these are unrecoverable
            if llm_response.status_code == 422:
                logger.info(
                    f"Response was not ok (status 422: {llm_response.text}), "
                    f"but will never be successful."
                )
                return llm_response

            # Check if response is successful
            if llm_response.ok:
                return llm_response

            # If we get here, the response was not ok but might be recoverable
            logger.info(
                f"Response was not ok (status {llm_response.status_code}: {llm_response.text}). Retrying ..."
            )

        except requests.exceptions.Timeout:
            logger.info("Request timed out, retrying ...")
        except requests.exceptions.ConnectionError:
            logger.info("Connection error, retrying ...")
        except requests.exceptions.RequestException as e:
            logger.info(f"Request failed with error: {str(e)}, retrying ...")
        except Exception as e:
            logger.info(f"Unknown error: {str(e)}, retrying ...")

        # Implement square root backoff
        retry_count += 1
        if retry_count <= REQUEST_MAX_RETRIES:
            # Square root backoff
            sleep_time = backoff_time * (retry_count**0.5)
            logger.info(
                f"Waiting {sleep_time:.3f} seconds before retry {retry_count}/{REQUEST_MAX_RETRIES} ..."
            )
            time.sleep(sleep_time)
        else:
            logger.error(
                f"Max retries ({REQUEST_MAX_RETRIES}) exceeded. Request failed."
            )

    # Return an empty response if the max number of retries has been reached.
    response = requests.Response()
    response.status_code = 503  # required to set a status code to check `.ok`
    return response
