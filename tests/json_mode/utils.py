from loguru import logger
import requests
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

openai_api_key = "-"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

sim_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def call_vllm_complete(
    prompts: list[list[dict[str, str]]],
    llm_name: str,
    options: dict[str, float | dict],
) -> requests.Response:
    response_format = options.get("response_format", None)
    if response_format is not None:
        if response_format.get("type") != "json":
            raise ValueError(
                "Only 'json' response format is supported in this test.")
    json_schema = response_format.get("json_schema", None)

    responses = []

    for i in range(len(prompts)):
        chat_response = client.chat.completions.create(
            model=llm_name,
            messages=prompts[i],
            temperature=options.get("temperature", 0.0),
            extra_body={"guided_json": json_schema} if json_schema else {},
        )
        responses.append(chat_response)

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
