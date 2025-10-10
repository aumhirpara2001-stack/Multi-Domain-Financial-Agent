import os, time, json
import pandas as pd
from openai import OpenAI

# initialize openai client with api key from environment
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# load financial qa dataset and keep only question/answer columns
df = pd.read_csv("datasets/Financial-QA-10k.csv")[["question", "answer"]]
df = df.dropna().reset_index(drop=True)
# limit to first 25 rows for testing
df = df.head(100)

def ask_model(question, model="gpt-5", temperature=0):
    """Ask one question → return raw string answer."""

    # create simple prompt asking for direct answer
    prompt = f"Answer the question. Return only the final answer.\n\nQuestion: {question}"
    try:
        # call openai api to get model response
        resp = client.responses.create(
            model="gpt-5",
            input=prompt,
            reasoning={"effort": "low"},
            text={"verbosity": "low"},
        )
        return resp.output_text.strip()
    except Exception as e:
        print("Error:", e)
        return None

def judge_answer(question, gold, pred, model="gpt-5", temperature=0):
    """Strict 0/0.5/1 rubric comparison."""
    # create prompt for ai judge to score model answers
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
        # call openai api to get judge evaluation
        resp = client.responses.create(
            model="gpt-5",
            input=judge_prompt,
            reasoning={"effort": "low"},
            text={"verbosity": "low"},
        )
        out = resp.output_text.strip()
        
        # extract json from markdown code block if present
        if out.startswith('```json'):
            # remove ```json from start and ``` from end
            json_str = out[7:-3].strip()  # remove ```json and ```
        elif out.startswith('```'):
            # remove ``` from start and end
            json_str = out[3:-3].strip()  # remove ``` and ```
        else:
            json_str = out
        
        # parse json response to get score and rationale
        data = json.loads(json_str)
        return data.get("score", 0), data.get("rationale", "")
    except Exception as e:
        print("Judge error:", e)
        return None, "parse_error"

# main evaluation loop
results = []
for i, row in df.iterrows():
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
        pd.DataFrame(results).to_csv("runs/base_eval_partial.csv", index=False)
    # time.sleep(0.5)  # small delay to stay under rate limits

# calculate and display final results
df_res = pd.DataFrame(results)
print("Mean accuracy:", df_res["score"].mean())
print(df_res.groupby("score").size())
# save final results to csv
df_res.to_csv("runs/base_eval_final.csv", index=False)
