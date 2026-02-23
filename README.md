# AI Portfolio Manager

An advanced, ML-powered Intelligent Portfolio Tracker built with Streamlit and Python. This application allows users to track their investments in real-time while receiving dynamic, AI-generated guidance from Anthropic's Claude to optimize their portfolio structure against their personal risk tolerance.

## 🚀 Key Features

- **Real-Time Market Data**: Live scraping of stocks, ETFs, and cryptocurrencies using the Yahoo Finance (`yfinance`) API.
- **AI Rebalancing Advisor (Anthropic Claude)**: Dynamically reads your portfolio and provides actionable, "Explain Like I'm 10"-friendly advice on what trades you need to make to re-align your holdings with your goals.
- **Machine Learning Macro Radar**: Utilizes a trained scikit-learn pipeline (`macro_risk_model.joblib`) to analyze rolling macroeconomic indicators (S&P 500 Volatility, VIX Z-Score, 10-Year Treasury yield changes) and probabilistically predict broader market corrections.
- **NLP News Sentiment Engine**: Scrapes live headlines for your specific holdings and scores them using a custom-trained Logistic Regression & TF-IDF pipeline (`sentiment_pipeline.joblib`) to measure market noise (+1 to -1 scale). 
- **Visual Gap Analysis**: State-of-the-art gap analysis tracking bars that instantly show if your Asset Class, Regional, or Sector balances are underweight or overweight.
- **Community Portfolio Benchmarking**: Compares your custom allocation against industry-standard templates (e.g., Ray Dalio's All-Weather, Classic 60/40, Yale Endowment).

## 🛠️ Technology Stack

- **Frontend Application**: Streamlit
- **Data Visualizations**: Plotly Graph Objects / Express
- **Financial Data Sources**: `yfinance`, Pandas
- **Machine Learning Infrastructure**: `scikit-learn`, `joblib`, `numpy`
- **Large Language Model API**: Anthropic Claude (`anthropic` python SDK)

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/portfolio-tracker.git
   cd portfolio-tracker
   ```

2. **Set up a Virtual Environment (Recommended):**
   ```bash
   python -m venv .venv_new
   source .venv_new/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory and add your secret API keys:
   ```env
   ANTHROPIC_API_KEY=your_claude_api_key_here
   ```

## ⚙️ Usage

To start the application locally, run:

```bash
streamlit run app.py
```

The app will become available in your browser at `http://localhost:8501`.

### Pre-Trained Models
The repository includes two custom ML training scripts in the `ml_pipeline/` directory.

- `train_model.py`: Generates the macro risk prediction model.
- `train_sentiment_model.py`: Generates the NLP headline classification pipeline.

> **Note**: These files output`.joblib` artifacts that the application loads natively upon launch! Ensure you run these scripts locally first if the `ml_pipeline/*.joblib` files are missing in your environment.

## 🗂️ Project Architecture

```
├── app.py                      # Main Streamlit Application Entrypoint
├── .env                        # Private API configuration
├── requirements.txt            # Python dependencies
├── components/                 
│   └── sidebar.py              # User Settings, Holdings Input, Profile creation
├── ml_pipeline/                
│   ├── train_model.py          # Macro Risk Random Forest Trainer
│   ├── train_sentiment_model.py# NLP Sentiment Pipeline Trainer
│   ├── macro_risk_model.joblib # Serialized Macro logic
│   └── sentiment_pipeline.joblib# Serialized NLP logic
├── utils/                      
│   ├── ai_advisor.py           # Anthropic API orchestration layer
│   ├── data_fetcher.py         # YFinance data, news, and meta scrapers
│   └── portfolio_metrics.py    # Math module (Vol, Beta, HHI calculation)
└── views/                      
    ├── dashboard.py            # Overview Tab logic
    ├── macro_radar.py          # Market Environment Tab logic
    ├── news.py                 # Sentiment Tab logic
    ├── popular_portfolios.py   # Community Benchmarking Tab logic
    └── suggestions.py          # Gap Analysis Rebalancing Tab logic
```

## 🛡️ License
MIT License. Feel free to clone, edit, and experiment!
