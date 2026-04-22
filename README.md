# AI Portfolio Advisor

A full-stack portfolio analytics platform built with Streamlit, combining real-time market data with multiple AI/ML integration patterns and a unified cross-component signal pipeline.

## Key Features

- **Real-Time Market Data**: Live prices for stocks, ETFs, and cryptocurrencies via Yahoo Finance (`yfinance`).
- **6 LLM Integration Patterns** (Anthropic Claude):
  1. **Single-call summarisation** (Haiku) — Portfolio advice, news summaries, macro analysis
  2. **Agentic tool-use loop** (Sonnet) — Chatbot with 6 tools that Claude calls autonomously across multiple turns
  3. **Structured JSON generation** (Sonnet) — Stress-test scenario parameters generated as validated JSON
  4. **Orchestrated multi-step decision chain** (Sonnet) — AI Autopilot: Claude proposes trades → Python simulates → Claude synthesises recommendation — enriched with macro, sentiment, efficient frontier, stress test, and backtest signals
  5. **Event-driven autonomous news agent** (Sonnet) — *Political Alpha*: a tool-use loop polling **live** feeds (GDELT 2.0, Truth Social archive, Reuters/CNBC RSS, plus yfinance per-holding news) and firing on tracked personas (Fed, ECB, presidents, macro actors). Scores sentiment with our own TF-IDF model, consults historical base rates, emits a structured trade proposal or a reasoned SKIP. Every decision logged to a SQLite ledger. A separate Historical Evaluation tab replays a curated 2018-2021 archive (real Reuters/Fed/ECB/Twitter headlines with ground-truth outcomes) to score agent accuracy with real PnL.
  6. **Multi-agent debate + extended thinking** (Haiku analysts + Opus judge) — three specialised analyst personas (BULL / BEAR / MACRO) each emit a JSON-schema-constrained argument; an Opus 4.x CIO judge with `thinking={"type":"enabled","budget_tokens":6000}` weighs all three and issues a `BULL / BEAR / NEUTRAL` verdict plus an actionable next step. System prompts use `cache_control: ephemeral` for cache reuse across topics.
- **Cross-Domain Pattern Gallery** — the same patterns applied outside finance to prove transferability: healthcare triage (pattern #3), contract clause review (pattern #1), travel itinerary planner with mocked tools (pattern #2). Every domain view renders a domain-specific disclaimer banner.
- **MCP Integration (Model Context Protocol)** — a separate `mcp_server/` process exposes the portfolio toolkit over stdio so Claude Desktop can call `get_portfolio_summary`, `calculate_portfolio_risk`, `what_if_trade`, `get_news`, and `get_political_alpha_ledger` autonomously. See [`mcp_server/CLAUDE_DESKTOP_CONFIG.md`](mcp_server/CLAUDE_DESKTOP_CONFIG.md).
- **2 Custom ML Models** (trained via GridSearchCV pipelines):
  - **Macro Risk Model** — Best-of-4 classifier (Logistic Regression, Random Forest, Gradient Boosting, XGBoost) selected by AUC-ROC via 5-fold `TimeSeriesSplit`. Trained on 26 years of macro data (12 features from 4 instruments) with balanced class handling.
  - **Sentiment Model** — Logistic Regression + TF-IDF with GridSearchCV over C, max_features, and n-gram range. Trained on ~6,000 financial tweets using time-ordered sequential split to prevent temporal leakage.
- **Cross-Component Signal Pipeline**: ML macro predictions, NLP sentiment scores, efficient frontier optimal weights, stress test vulnerabilities, and backtest performance all flow into a shared signal registry (`st.session_state`) — enriching every LLM prompt with quantitative context.
- **Quantitative Rigor**: Dynamic risk-free rate (live US 10Y Treasury yield), CVaR (Conditional Value at Risk), Sharpe, Beta, HHI, Max Drawdown. Model staleness tracking with automated warnings.
- **13 Interactive Views**: Dashboard, AI Chat, Political Alpha, Debate (CIO), Cross-Domain Gallery, Stress Test, Suggestions (Autopilot), Macro Radar, News & Sentiment, Efficient Frontier, Backtest, Standard Portfolios, ML Transparency.

## Technology Stack

| Layer | Technologies |
|-------|-------------|
| Frontend | Streamlit |
| Visualizations | Plotly Graph Objects / Express |
| Financial Data | `yfinance`, Pandas |
| ML / NLP | `scikit-learn`, `xgboost`, `joblib`, `scipy`, `numpy` |
| LLM API | Anthropic Claude (`anthropic` Python SDK) — Haiku + Sonnet |
| Feature Engineering | Centralized in `ml_pipeline/features.py` (shared between training & inference) |

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/paugratacosfuste/portfolio_manager.git
   cd portfolio_manager
   ```

2. **Set up a Virtual Environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   ANTHROPIC_API_KEY=your_claude_api_key_here
   ```

## Usage

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

### Training the ML Models

The repository includes pre-trained `.joblib` artifacts. To retrain with fresh data:

```bash
# Macro Risk Model (downloads data up to today automatically)
PYTHONPATH=. python ml_pipeline/train_model.py

# Sentiment Model
PYTHONPATH=. python ml_pipeline/train_sentiment_model.py
```

Both scripts output `.joblib` model files and evaluation artifacts that the app loads at runtime. The macro model automatically fetches data up to the current date — no hardcoded end dates.

## Project Architecture

```
├── app.py                         # Main Streamlit entrypoint & page routing
├── components/
│   └── sidebar.py                 # User profile, holdings input, AI settings
├── ml_pipeline/
│   ├── features.py                # Centralized macro feature engineering (shared train/inference)
│   ├── train_model.py             # Macro Risk Model trainer (4-model GridSearchCV comparison)
│   ├── train_sentiment_model.py   # Sentiment Model trainer (TF-IDF + LR GridSearchCV)
│   ├── macro_risk_model.joblib    # Serialized macro classifier pipeline
│   ├── macro_eval_results.joblib  # Evaluation artifacts (metrics, CV scores, importances)
│   ├── sentiment_pipeline.joblib  # Serialized NLP pipeline
│   └── sentiment_eval_results.joblib # Sentiment evaluation artifacts
├── utils/
│   ├── ai_advisor.py              # LLM orchestration (advice, news summary, autopilot chain)
│   ├── chatbot_tools.py           # Agentic chatbot tool schemas & executors (6 tools)
│   ├── data_fetcher.py            # yfinance data, news, and metadata scrapers
│   ├── portfolio_metrics.py       # Risk math (volatility, beta, HHI, Sharpe, CVaR, drawdown)
│   └── stress_test.py             # Stress test scenario simulation engine
├── views/
│   ├── dashboard.py               # Portfolio overview with risk metrics
│   ├── ai_chat.py                 # Agentic chatbot (tool-use loop)
│   ├── stress_test.py             # AI-generated stress scenarios with simulation
│   ├── suggestions.py             # Autopilot: orchestrated multi-step LLM chain
│   ├── macro_radar.py             # ML macro prediction with indicator cards
│   ├── news.py                    # NLP sentiment scoring with portfolio-weighted aggregation
│   ├── efficient_frontier.py      # Monte Carlo simulation & MPT optimization
│   ├── backtest.py                # Historical portfolio backtesting vs SPY
│   ├── popular_portfolios.py      # Standard portfolio benchmarking
│   └── ml_transparency.py         # Full model evaluation (ROC, PR, calibration, CV, learning curves)
└── styles/
    └── main.css                   # Custom styling
```

## Cross-Component Signal Pipeline

The platform uses `st.session_state` as a shared signal registry. Each view computes and stores its outputs:

| Signal | Source View | Consumers |
|--------|-----------|-----------|
| `macro_prediction` | Macro Radar | Suggestions, AI Chat, Autopilot LLM prompts |
| `portfolio_sentiment` | News & Sentiment | Suggestions, Autopilot LLM prompts |
| `optimal_weights` | Efficient Frontier | Suggestions, Autopilot LLM prompts |
| `last_stress_test` | Stress Test | Suggestions, Autopilot LLM prompts |
| `backtest_metrics` | Backtest | Suggestions, Autopilot LLM prompts |

This ensures every LLM prompt is enriched with the latest quantitative context from all components.

## ML Transparency

The **ML Transparency** tab provides full model evaluation including:
- Classification reports, confusion matrices
- ROC curves and Precision-Recall curves with contextual interpretation
- Calibration curves (reliability diagrams)
- Cross-validation results (5-fold TimeSeriesSplit)
- Learning curves
- Feature importances (Gini + Permutation)
- Feature correlation heatmaps
- Live model introspection (hyperparameters, scaler statistics, top sentiment words)

All metrics are computed live from saved evaluation artifacts — nothing is hardcoded.

## License

MIT License.
