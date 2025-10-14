# Benchmarking Script

This script evaluates how well AI models answer financial questions by comparing their responses to expert-provided answers (in this case it's specifically the `Financial-QA-10k` set)

## What it does

1. Loads a dataset
2. Sends each question to an AI model (in this case an OpenAI GPT-5 model, but will adjust to ReAct agentic models for future) to get predictions
3. Uses another AI model (GPT-5) as a judge to score how well the predictions match the expert answers
4. Saves results showing accuracy scores and explanations

## Scoring system

- **1.0**: Answer is numerically or semantically exact
- **0.5**: Method is correct with minor errors (rounding, wording differences)
- **0.0**: Answer is incorrect

## Files

- `run.py` - Main evaluation script
- `datasets/Financial-QA-10k.csv` - Question and answer dataset
- `runs/base_eval_final.csv` - Complete results
- `runs/base_eval_partial.csv` - Progress checkpoints (saves every 20 questions)

## Setup

1. Set your OpenAI API key: `$env:OPENAI_API_KEY="your-key-here"`
2. Run: `python run.py`

## Current results

The script is currently set to test 25 questions. Feel free to change this. The line is `df = df.head(25)`. Results show mean accuracy and score distribution across all questions.

## Notes
- Saves progress periodically in case of interruptions in `runs\base_eval_partial.csv`
- Judge provides brief explanations for each score
# Benchmarking Script

This script evaluates how well AI models answer financial questions by comparing their responses to expert-provided answers (Financial‑QA‑10k).

## What it does
1. Loads the dataset.
2. Sends each question to an **Answer model** to get a prediction.
   - Default: Together AI Llama‑4 (`meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`).
   - Optional: OpenAI GPT‑5 if you disable Together.
3. Uses **GPT‑5** (OpenAI) as a **Judge** to score how well the prediction matches the expert answer.
4. Saves per‑question results and prints summary metrics.

## Scoring system
- **1.0**: numerically or semantically exact  
- **0.5**: method is correct with a minor slip (e.g., rounding, small wording)  
- **0.0**: incorrect

## Files
- `run.py` — main evaluation script
- `datasets/Financial-QA-10k.csv` — question and answer dataset
- `runs/base_eval_partial_llama.csv` — periodic checkpoints (every 20 questions)
- `runs/base_eval_final_llama.csv` — final results for the run
- `graph.py` — plots score distribution (plotnine); default output: `runs/llama_score_distribution_plotnine.png`

## Setup
Install:
```bash
python -m pip install -U together openai pandas plotnine
```

Environment:
```bash
# required
export OPENAI_API_KEY="..."         # used for the GPT-5 judge
export TOGETHER_API_KEY="..."       # used for the Together answer model

# optional
export TOGETHER_MODEL="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
export USE_TOGETHER=1   # 1 = use Together Llama-4 for answers (default), 0 = use GPT-5 for answers
```

## Run
```bash
python run.py
```
Defaults:
- Uses the first 100 questions (`df = df.head(100)`).
- Prints the active answer model and the judge model at the start.
- Saves progress every 20 questions.
- Writes the final CSV to `runs/base_eval_final_llama.csv`.

## Notes
- The **Answer model** uses only the `temperature` flag (default 0).  
- The **Judge (GPT‑5)** does **not** accept `temperature`; the code omits it.  
- You can switch answering to GPT‑5 by setting `USE_TOGETHER=0`.