import json

output_file = "financebench_open_source.jsonl"

questions = [
    # 📊 Corporate Finance
    ("How is Return on Equity (ROE) calculated?", "ROE = Net Income ÷ Shareholder's Equity. It measures profitability relative to equity invested."),
    ("What does a P/E ratio of 35x suggest?", "Investors are paying $35 for every $1 of earnings, often reflecting growth expectations or overvaluation."),
    ("How is Weighted Average Cost of Capital (WACC) calculated?", "WACC = (E/V × Re) + (D/V × Rd × (1 – Tc)), where E = equity, D = debt, V = total capital, Re = cost of equity, Rd = cost of debt, Tc = tax rate."),
    ("How do you calculate Enterprise Value (EV)?", "EV = Market Cap + Total Debt – Cash & Equivalents."),

    # 🧮 Accounting
    ("What is the difference between accrual accounting and cash accounting?", "Accrual accounting records revenues and expenses when earned or incurred, while cash accounting records them when cash changes hands."),
    ("How is depreciation expense calculated using straight-line method?", "Depreciation = (Cost – Salvage Value) ÷ Useful Life."),

    # 📈 Quantitative Finance
    ("What is Brownian motion in finance?", "Brownian motion models continuous random price movements and underpins stochastic calculus."),
    ("How is a random walk different from Brownian motion?", "A random walk is discrete, while Brownian motion is continuous. Both model unpredictable asset paths."),
    ("What is Monte Carlo simulation used for in risk management?", "Monte Carlo simulates thousands of random scenarios to estimate distributions of outcomes like portfolio returns or VaR."),

    # 🧬 Mathematical Finance
    ("What is Ito's Lemma used for?", "Ito's Lemma derives the dynamics of functions of stochastic processes, essential in option pricing models."),
    ("How is the Black-Scholes formula structured?", "C = S0N(d1) – Ke^(–rt)N(d2), where d1 and d2 are functions of volatility, time, and interest rate."),

    # 🧑‍🎓 Portfolio Theory
    ("What is the efficient frontier?", "The efficient frontier represents portfolios that maximize expected return for a given level of risk."),
    ("How is the Sharpe ratio calculated?", "Sharpe ratio = (Portfolio Return – Risk-Free Rate) ÷ Portfolio Standard Deviation."),

    # 🧑‍⚖️ Regulatory & Reporting
    ("What is a 10-K filing?", "A 10-K is an annual report filed with the SEC by public companies, detailing financial performance."),
    ("What does SOX compliance require?", "SOX mandates internal controls, auditor independence, and CEO/CFO certification of financial statements."),

    # 🧠 Behavioral Finance
    ("What is loss aversion?", "Loss aversion is the tendency for people to prefer avoiding losses over acquiring equivalent gains."),
    ("What is prospect theory?", "Prospect theory explains how people make decisions under risk, valuing losses more heavily than gains."),
]

with open(output_file, "w", encoding="utf-8") as f:
    for i in range(10000):  # target size
        q, a = questions[i % len(questions)]  # cycle through or expand
        entry = {
            "question": q,
            "answer": a,
            "id": f"synthetic_{i+1:05d}",
            "source_dataset": "synthetic_finance_council"
        }
        f.write(json.dumps(entry) + "\n")

print("✅ Generated 10,000 synthetic entries")
