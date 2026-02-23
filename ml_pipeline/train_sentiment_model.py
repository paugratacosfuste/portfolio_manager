import os
import zipfile
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

def train_sentiment_model(zip_filename="ml_pipeline/stock_market_sentiment.zip"):
    print("=" * 70)
    print("STOCK SENTIMENT NLP MODEL TRAINING")
    print("=" * 70)

    df = None
    if os.path.exists(zip_filename):
        print(f"Found '{zip_filename}'. Reading CSV...")
        try:
            with zipfile.ZipFile(zip_filename, 'r') as z:
                # Find the first csv file
                csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                if not csv_files:
                    raise FileNotFoundError("No CSV file found inside ZIP.")
                
                with z.open(csv_files[0]) as f:
                    df = pd.read_csv(f)
                    print(f"Loaded extracted CSV data from memory.")
        except Exception as e:
            print(f"Failed to read from zip: {e}")
            return
    else:
        print(f"Zip file '{zip_filename}' not found.")
        return

    try:
        text_col = next(col for col in df.columns if 'text' in col.lower() or 'tweet' in col.lower())
        target_col = next(col for col in df.columns if 'sentiment' in col.lower() or 'target' in col.lower() or 'score' in col.lower())

        if df[target_col].dtype == 'object':
            unique_vals = sorted(df[target_col].unique())
            mapping = {val: idx - (len(unique_vals)//2) for idx, val in enumerate(unique_vals)}
            df['sentiment_mapped'] = df[target_col].map(mapping)
            y = df['sentiment_mapped']
        else:
            y = df[target_col]

        X = df[text_col].astype(str)
        print(f"Loaded {len(df)} entries.")
    except Exception as e:
        print(f"Error parsing dataset: {e}")
        return

    print("\n--- Model Training ---")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("1. Training Pipeline (TF-IDF + Logistic Regression)...")
    pipe = Pipeline([
        ('vectorizer', TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2), min_df=2)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42, C=1.0, class_weight='balanced'))
    ])

    pipe.fit(X_train, y_train)

    print("2. Evaluating Model on Test Set...")
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    os.makedirs('ml_pipeline', exist_ok=True)

    print("\n3. Saving Pipeline Models using Joblib...")
    model_path = 'ml_pipeline/sentiment_pipeline.joblib'
    joblib.dump(pipe, model_path)
    print(f"Sentiment Pipeline successfully saved to {model_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        train_sentiment_model(sys.argv[1])
    else:
        train_sentiment_model()
