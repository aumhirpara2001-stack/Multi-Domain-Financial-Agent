import os, time, json
import pandas as pd
from openai import OpenAI
from together import Together

USE_TOGETHER = os.getenv("USE_TOGETHER", "1") == "1"
client_openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
client_together = Together(api_key=os.environ.get("TOGETHER_API_KEY", ""))

# load financial qa dataset and keep only question/answer columns
df = pd.read_csv("datasets/Financial-QA-10k.csv")[["question", "answer"]]
df = df.dropna().reset_index(drop=True)
# limit to first 25 rows for testing
df = df.head(100)

def ask_model(question, model=None, temperature=0):
    """Ask one question → return raw string answer."""

    if USE_TOGETHER:
        # default to Llama‑4 unless overridden
        model = model or os.getenv("TOGETHER_MODEL", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")
        prompt = f"Answer the question. Return only the final answer.\n\nQuestion: {question}"
        try:
            resp = client_together.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,  # only flag we set
            )
            # Together chat completions shape
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print("Error:", e)
            return None
    else:
        model = model or "gpt-5"
        prompt = f"Answer the question. Return only the final answer.\n\nQuestion: {question}"
        try:
            resp = client_openai.responses.create(
                model=model,
                input=prompt,
                temperature=temperature,  # only flag we set
            )
            if hasattr(resp, "output_text") and resp.output_text:
                return resp.output_text.strip()
            if getattr(resp, "choices", None):
                msg = resp.choices[0].message
                content = msg["content"] if isinstance(msg, dict) else getattr(msg, "content", "")
                return (content or "").strip()
            return None
        except Exception as e:
            print("Error:", e)
            return None

def judge_answer(question, gold, pred, model="gpt-5"):
    """Strict 0/0.5/1 rubric comparison."""

    judge_prompt = f"""
You are a strict grader. Compare the model's answer to the gold answer.

Question: {question}
Gold answer: {gold}
Model answer: {pred}

Rules:
- 1 if numerically or semantically exact.
- 0.5 if method is right with a minor slip (≤1% rounding or small wording difference).
- 0 otherwise.

Return JSON only in this format:
{{"score": 0|0.5|1, "rationale": "<≤20 words>"}}
"""
    try:
        resp = client_openai.responses.create(
            model=model,
            input=judge_prompt,
            # temperature=temperature,
        )
        if hasattr(resp, "output_text") and resp.output_text:
            out = resp.output_text.strip()
        elif getattr(resp, "choices", None):
            msg = resp.choices[0].message
            out = (msg["content"] if isinstance(msg, dict) else getattr(msg, "content", "")).strip()
        else:
            out = ""

        if out.startswith('```json'):
            json_str = out[7:-3].strip()
        elif out.startswith('```'):
            json_str = out[3:-3].strip()
        else:
            json_str = out

        data = json.loads(json_str)
        return data.get("score", 0), data.get("rationale", "")
    except Exception as e:
        print("Judge error:", e)
        return None, "parse_error"

# main evaluation loop
results = []
for i, row in df.iterrows():
    if i == 0:
        answer_model = (
            os.getenv("TOGETHER_MODEL", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")
            if USE_TOGETHER else
            "gpt-5"
        )
        provider = "Together" if USE_TOGETHER else "OpenAI"
        print(f"Answer model: {answer_model} via {provider}")
        print("Judge model: gpt-5 via OpenAI")
    q, gold = row["question"], row["answer"]

    # get model prediction for this question
    pred = ask_model(q)
    # get ai judge score and rationale
    score, why = judge_answer(q, gold, pred)

    # store results for this question
    results.append({
        "id": i,
        "question": q,
        "gold": gold,
        "pred": pred,
        "score": score,
        "why": why
    })

    # save progress every 20 questions
    if i % 20 == 0:
        print(f"{i}/{len(df)} done...")
        pd.DataFrame(results).to_csv("runs/base_eval_partial_llama.csv", index=False)
    # time.sleep(0.5)  # small delay to stay under rate limits

# calculate and display final results
df_res = pd.DataFrame(results)
print("Mean accuracy:", df_res["score"].mean())
print(df_res.groupby("score").size())
# save final results to csv
df_res.to_csv("runs/base_eval_final_llama.csv", index=False)
