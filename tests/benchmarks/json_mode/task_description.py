"""Description of evaluation tasks (instructions + expected schema definition)"""

from pydantic import BaseModel


class GenerateAnswer(BaseModel):
    answer: str


class RateContext(BaseModel):
    context_score: int


class AssessAnswerability(BaseModel):
    answerable_question: bool


class ParaphraseQuestions(BaseModel):
    paraphrased_questions: list[str]


class RAGAS(BaseModel):
    faithfulness_score: float
    answer_relevance_score: float
    context_relevance_score: float


class GenerateAnswerWithConfidence(BaseModel):
    answer: str
    confidence: int


class GenerateAnswersWithConfidence(BaseModel):
    answers: list[GenerateAnswerWithConfidence]


TASK_DESCRIPTIONS = {
    "GenerateAnswer": {
        "task_instructions":
        "Assess the context and answer the question. If "
        "the context does not contain sufficient information to answer "
        'the question, respond with "NOT ENOUGH CONTEXT".',
        "response_format":
        '{"answer": "string"}',
    },
    "RateContext": {
        "task_instructions":
        "Assess how well the context helps answer the question.",
        "response_format": '{"context_score": "int (0-5)"}',
    },
    "AssessAnswerability": {
        "task_instructions": "Determine if the question is answerable based "
        "on the context.",
        "response_format": '{"answerable_question": "bool"}',
    },
    "ParaphraseQuestions": {
        "task_instructions":
        "Generate 3 paraphrased versions of the given question.",
        "response_format":
        '{"paraphrased_questions": ["string", "string", "string"]}',
    },
    "RAGAS": {
        "task_instructions":
        "Assess the faithfulness, answer relevance, and "
        "context relevance given a question, context, and answer.",
        "response_format":
        '{"faithfulness_score": "float (0-5)", '
        '"answer_relevance_score": "float (0-5)", '
        '"context_relevance_score": "float (0-5)"}',
    },
    "GenerateAnswerWithConfidence": {
        "task_instructions":
        "Generate an answer with a confidence score."
        "If the context does not contain sufficient information to answer "
        'the question, respond with "NOT ENOUGH CONTEXT".',
        "response_format":
        '{"answer": "string", "confidence": "int (0-5)"}',
    },
    # Super slow in the benchmarks, so commented out for now
    # "GenerateAnswersWithConfidence": {
    #     "task_instructions":
    #     "Generate multiple answers with confidence scores."
    #     "If the context does not contain sufficient information to answer "
    #     'the question, respond with "NOT ENOUGH CONTEXT".',
    #     "response_format":
    #     '{"answers" : [{"answer": "string", "confidence": "int (0-5)"}, ...]}',
    # },
}

TASK_NAME_TO_TASK_SCHEMA = {
    "GenerateAnswer": GenerateAnswer,
    "RateContext": RateContext,
    "AssessAnswerability": AssessAnswerability,
    "ParaphraseQuestions": ParaphraseQuestions,
    "RAGAS": RAGAS,
    "GenerateAnswerWithConfidence": GenerateAnswerWithConfidence,
    "GenerateAnswersWithConfidence": GenerateAnswersWithConfidence,
}
