"""LLM-as-a-Judge evaluation for RAG system."""

import os
import json
from typing import Tuple, Optional
import openai
from dotenv import load_dotenv

load_dotenv()


def llm_as_judge(
    question: str,
    gold_answer: str,
    predicted_answer: str,
    judge_model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    use_openai: bool = False
) -> Tuple[Optional[float], str]:
    """
    Use an LLM to evaluate the quality of a predicted answer.

    Args:
        question: The original question
        gold_answer: The ground truth answer
        predicted_answer: The model's predicted answer
        judge_model: Model to use for judging
        use_openai: If True, use OpenAI API; otherwise use Together API

    Returns:
        Tuple of (score, justification) where:
            - score: Float from 0.0 to 1.0, or None if evaluation failed
            - justification: String explaining the score
    """
    if not predicted_answer or not gold_answer:
        return 0.0, "Empty answer"

    prompt = f"""You are a financial QA evaluation assistant. Your task is to evaluate how well a predicted answer matches the ground truth answer.

Question: {question}

Ground Truth Answer: {gold_answer}

Predicted Answer: {predicted_answer}

Evaluate the predicted answer on a scale from 0.0 to 1.0 based on:
1. **Correctness** - Does it contain the same factual information as the ground truth?
2. **Completeness** - Does it include all key details from the ground truth?
3. **Relevance** - Does it directly answer the question asked?

Scoring guidelines:
- 1.0: Perfect match - contains all correct information
- 0.8-0.9: Very good - minor differences in wording but same meaning
- 0.6-0.7: Good - correct main point but missing some details
- 0.4-0.5: Partial - some correct information but significant gaps
- 0.2-0.3: Poor - mostly incorrect or incomplete
- 0.0-0.1: Wrong - completely incorrect or irrelevant

Return ONLY valid JSON in this exact format (no other text):
{{"score": 0.XX, "justification": "Brief explanation of the score"}}"""

    try:
        if use_openai:
            # Use OpenAI API
            client = openai.OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY")
            )
        else:
            # Use Together API
            client = openai.OpenAI(
                api_key=os.environ.get("TOGETHER_API_KEY"),
                base_url="https://api.together.xyz/v1"
            )

        response = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200
        )

        content = response.choices[0].message.content.strip()

        # Parse JSON response
        # Handle potential markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        result = json.loads(content)
        score = float(result.get("score", 0.0))
        justification = result.get("justification", "")

        # Ensure score is in valid range
        score = max(0.0, min(1.0, score))

        return score, justification

    except json.JSONDecodeError as e:
        return None, f"Failed to parse judge response: {str(e)}"
    except Exception as e:
        return None, f"Judge evaluation error: {str(e)}"


def batch_llm_judge(
    questions: list,
    gold_answers: list,
    predicted_answers: list,
    judge_model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    use_openai: bool = False
) -> list:
    """
    Evaluate multiple predictions using LLM-as-a-Judge.

    Args:
        questions: List of questions
        gold_answers: List of ground truth answers
        predicted_answers: List of predicted answers
        judge_model: Model to use for judging
        use_openai: If True, use OpenAI API; otherwise use Together API

    Returns:
        List of tuples (score, justification) for each prediction
    """
    results = []

    for q, gold, pred in zip(questions, gold_answers, predicted_answers):
        score, justification = llm_as_judge(
            question=q,
            gold_answer=gold,
            predicted_answer=pred,
            judge_model=judge_model,
            use_openai=use_openai
        )
        results.append((score, justification))

    return results
