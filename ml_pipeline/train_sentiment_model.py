import os
import zipfile
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
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

    # Use sequential split (first 80% train, last 20% test) to prevent
    # temporal leakage. Financial text shares regime context within time periods.
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"Time-ordered split: {len(X_train)} train, {len(X_test)} test")

    print("1. Training Pipeline (TF-IDF + Logistic Regression with GridSearch)...")

    from sklearn.model_selection import GridSearchCV
    pipe = Pipeline([
        ('vectorizer', TfidfVectorizer(stop_words='english', min_df=2)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
    ])

    param_grid = {
        'vectorizer__max_features': [3000, 5000, 10000],
        'vectorizer__ngram_range': [(1, 1), (1, 2)],
        'classifier__C': [0.1, 0.5, 1.0, 5.0],
    }

    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    pipe = grid.best_estimator_
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV accuracy: {grid.best_score_:.4f}")

    print("2. Evaluating Model on Test Set...")
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    # --- EXPANDED EVALUATION ARTIFACTS ---
    print("\n3. Computing expanded evaluation artifacts...")

    # Cross-validation scores with StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='accuracy')
    print(f"CV Accuracy scores: {cv_scores}")
    print(f"CV Accuracy mean: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # Learning curve
    train_sizes, train_scores, val_scores = learning_curve(
        pipe, X_train, y_train, cv=skf, scoring='accuracy',
        train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=-1
    )
    print(f"Learning curve computed with {len(train_sizes)} points")

    os.makedirs('ml_pipeline', exist_ok=True)

    print("\n4. Saving Pipeline Models using Joblib...")
    model_path = 'ml_pipeline/sentiment_pipeline.joblib'
    joblib.dump(pipe, model_path)
    print(f"Sentiment Pipeline successfully saved to {model_path}")

    # Save evaluation artifacts for ML Transparency visualizations
    import datetime as _dt
    eval_path = 'ml_pipeline/sentiment_eval_results.joblib'
    joblib.dump({
        'y_test': y_test.values,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'classes': pipe.classes_,
        'cv_scores': cv_scores,
        'train_sizes': train_sizes,
        'train_scores': train_scores,
        'val_scores': val_scores,
        # Training metadata
        'trained_at': _dt.datetime.now().isoformat(),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'accuracy': acc,
        'split_method': 'sequential (time-ordered)',
        'best_grid_params': grid.best_params_ if 'grid' in dir() else {},
    }, eval_path)
    print(f"Evaluation artifacts saved to {eval_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        train_sentiment_model(sys.argv[1])
    else:
        train_sentiment_model()
