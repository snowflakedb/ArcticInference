from loguru import logger
import requests
from openai import OpenAI

openai_api_key = "-" 
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def call_vllm_complete(
    prompts: list[list[dict[str, str]]],
    llm_name: str,
    options: dict[str, float | dict],
) -> requests.Response:
    response_format = options.get("response_format", None)
    if response_format is not None:
        if response_format.get("type") != "json":
            raise ValueError("Only 'json' response format is supported in this test.")
    json_schema = response_format.get("json_schema", None)

    responses = []

    for i in range(len(prompts)):
        chat_response = client.chat.completions.create(
            model=llm_name,
            messages=prompts[i],
            temperature=options.get("temperature", 0.0),
            extra_body={
                "guided_json": json_schema
            } if json_schema else {},
        )
        responses.append(chat_response)

    return responses

