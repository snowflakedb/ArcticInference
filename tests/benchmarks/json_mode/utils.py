import asyncio
import requests
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer, util

openai_api_key = "-"
openai_api_base = "http://localhost:8000/v1"

client = AsyncOpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

sim_model = SentenceTransformer(
    'sentence-transformers/all-MiniLM-L6-v2',
    device='cpu')


async def get_chat_completion(
    prompt: list[dict[str, str]],
    llm_name: str,
    temperature: float = 0.0,
    json_schema: dict[str, str] | None = None,
) -> requests.Response:
    response = await client.chat.completions.create(
        model=llm_name,
        messages=prompt,
        temperature=temperature,
        extra_body={"guided_json": json_schema})
    return response


async def call_vllm_complete(
    prompts: list[list[dict[str, str]]],
    llm_name: str,
    options: dict[str, float | dict],
) -> requests.Response:
    response_format = options.get("response_format", None)
    json_schema = response_format.get("schema", None)
    assert json_schema is not None

    temperature = options.get("temperature", 0.0)
    tasks = [
        get_chat_completion(prompt, llm_name, temperature, json_schema)
        for prompt in prompts
    ]

    responses = await asyncio.gather(*tasks)

    return responses


def compute_sentence_similarity(sentence_a: str, sentence_b: str) -> float:
    """
    Computes the cosine similarity between two sentences using a pre-trained SentenceTransformer model.
    
    Args:
        sentence_a (str): The first sentence.
        sentence_b (str): The second sentence.
    
    Returns:
        float: Cosine similarity score between the two sentences.
    """
    embedding_a = sim_model.encode(sentence_a, convert_to_tensor=True)
    embedding_b = sim_model.encode(sentence_b, convert_to_tensor=True)

    # Compute cosine similarity
    similarity_score = util.pytorch_cos_sim(embedding_a, embedding_b).item()

    return similarity_score
