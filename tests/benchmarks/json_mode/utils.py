import asyncio
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer, util

sim_model = SentenceTransformer(
    'sentence-transformers/all-MiniLM-L6-v2',
    device='cpu')


async def get_chat_completion(
    prompt: list[dict[str, str]],
    llm_name: str,
    client: AsyncOpenAI,
    options: dict,
):
    """Makes a single chat completion request using a provided client."""
    response = await client.chat.completions.create(
        model=llm_name,
        messages=prompt,
        **options,
    )
    return response


async def call_vllm_complete(
    prompts: list[list[dict[str, str]]],
    llm_name: str,
    options: dict[str, float | dict],
    port: int,
) -> list:
    client = AsyncOpenAI(
        api_key="-",
        base_url=f"http://localhost:{port}/v1",
    )

    tasks = [
        get_chat_completion(prompt, llm_name, client, options)
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