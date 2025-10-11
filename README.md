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