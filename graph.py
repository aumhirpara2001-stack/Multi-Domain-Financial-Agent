#!/usr/bin/env python3
# graph.py — plot score distribution for latest Llama run (judged by GPT-5)

import argparse, os, sys
import pandas as pd
from plotnine import (
    ggplot, aes, geom_bar, geom_text, scale_fill_manual,
    labs, theme_bw, theme, element_text, ggsave
)

def main():
    ap = argparse.ArgumentParser(description="Plot score distribution for Llama run (GPT-5 judge).")
    ap.add_argument("--csv", default="runs/base_eval_final_llama.csv",
                    help="Path to results CSV (default: runs/base_eval_final_llama.csv)")
    ap.add_argument("--out", default="runs/llama_score_distribution_plotnine.png",
                    help="Output image path (default: runs/llama_score_distribution_plotnine.png)")
    ap.add_argument("--title", default="Score Distribution – Llama-4-Maverick-17B-128E-Instruct-FP8",
                    help="Plot title")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"[graph.py] CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    if "score" not in df.columns:
        sys.exit(f"[graph.py] Column 'score' not found in {args.csv}")

    # numeric scores → ordered categorical strings "0.0", "0.5", "1.0"
    cat_order = ["0.0", "0.5", "1.0"]
    scores = pd.to_numeric(df["score"], errors="coerce").dropna().map(lambda x: f"{x:.1f}")
    scores = pd.Categorical(scores, categories=cat_order, ordered=True)

    counts_df = (
        pd.DataFrame({"score": scores})
        .groupby("score", observed=True).size()
        .reindex(cat_order, fill_value=0)
        .reset_index(name="n")
    )

    # consistent colors: red (0.0), orange (0.5), green (1.0)
    color_map = {"0.0": "red", "0.5": "orange", "1.0": "green"}

    p = (
        ggplot(counts_df, aes(x="score", y="n", fill="score"))
        + geom_bar(stat="identity")
        + geom_text(aes(label="n"), va="bottom", size=9)  # labels on bars
        + scale_fill_manual(values=[color_map[s] for s in cat_order], breaks=cat_order, name="Score")
        + labs(title=args.title, x="Score", y="Number of Questions")
        + theme_bw()
        + theme(
            plot_title=element_text(ha="center", size=12, weight="bold"),
            axis_title_x=element_text(size=10),
            axis_title_y=element_text(size=10),
            legend_title=element_text(size=9),
            legend_text=element_text(size=9),
        )
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ggsave(p, filename=args.out, dpi=160, width=6, height=4.5)

    # console summary
    print(f"[graph.py] Read:  {args.csv}")
    print(f"[graph.py] Saved: {args.out}")
    print(counts_df.rename(columns={"n": "count"}).to_string(index=False))

if __name__ == "__main__":
    main()
