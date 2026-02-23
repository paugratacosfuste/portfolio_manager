import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def generate_dummy_sentiment_data():
    """
    Generates a dummy dataset of stock-related tweets if the Kaggle dataset is not found.
    This ensures the pipeline builds correctly and doesn't block the UI development.
    """
    print("\n⚠️  Kaggle 'tweet_sentiment.csv' not found. Generating a dummy training dataset to initialize the model...")
    print("    To use the real Kaggle ML model, download the dataset, name it 'tweet_sentiment.csv', and put it in the root folder.")
    
    # 1 = Positive, 0 = Neutral, -1 = Negative
    texts = [
        "Apple announces record-breaking earnings for Q4, stock soars!", # +1
        "Microsoft cloud growth exceeds expectations, strong buy rating.", # +1
        "Nvidia launches new chip sequence, highly anticipated market dominator.", # +1
        "Tesla vehicle deliveries up 20% year over year.", # +1
        "Great dividend payout announced by JP Morgan today.", # +1
        "Apple stock trading flat ahead of earnings report.", # 0
        "Microsoft CEO holding a standard press conference today.", # 0
        "Market remains relatively stable despite mixed economic data.", # 0
        "Tesla updates software on older models.", # 0
        "No major news for tech stocks this morning, trading sideways.", # 0
        "Apple faces severe supply chain disruptions in Asia, shares down 4%.", # -1
        "Huge antitrust lawsuit filed against Microsoft in Europe.", # -1
        "Nvidia earnings miss estimates by a wide margin, stock plummets.", # -1
        "Tesla recalls over 100,000 vehicles due to safety defect.", # -1
        "Rising inflation crushes tech sector margins, widespread sell-off." # -1
    ] * 20 # Duplicate them to create enough samples for train_test_split
    
    labels = [1, 1, 1, 1, 1,  # Positives
              0, 0, 0, 0, 0,  # Neutrals
             -1, -1, -1, -1, -1] * 20 # Negatives
             
    return pd.DataFrame({'text': texts, 'sentiment': labels})

def train_sentiment_model(csv_path="tweet_sentiment.csv"):
    """
    Trains an NLP model (TF-IDF Vectorizer + Logistic Regression) on text data to predict sentiment.
    """
    print("=" * 70)
    print("STOCK SENTIMENT NLP MODEL TRAINING")
    print("=" * 70)
    
    if os.path.exists(csv_path):
        print(f"Loading '{csv_path}'...")
        # Assume the Kaggle CSV has columns like 'Tweet' and 'Sentiment'
        # Adjust these column names depending on the exact Kaggle CSV format
        try:
            df = pd.read_csv(csv_path)
            # Find the text column
            text_col = next(col for col in df.columns if 'text' in col.lower() or 'tweet' in col.lower())
            
            # Use 'sentiment' or matching target column
            target_col = next(col for col in df.columns if 'sentiment' in col.lower() or 'target' in col.lower() or 'score' in col.lower())
            
            # Map sentiments to -1, 0, 1 if they are categorical
            if df[target_col].dtype == 'object':
                 mapping = {label: idx-1 for idx, label in enumerate(np.unique(df[target_col]))} # Simplistic heuristic
                 df['sentiment_mapped'] = df[target_col].map(mapping)
                 y = df['sentiment_mapped']
            else:
                 y = df[target_col]
                 
            X = df[text_col].astype(str)
            print(f"Loaded {len(df)} entries from Kaggle dataset.")
                 
        except Exception as e:
            print(f"Error parsing Kaggle CSV: {e}. Falling back to dummy data.")
            df = generate_dummy_sentiment_data()
            X = df['text']
            y = df['sentiment']
    else:
        df = generate_dummy_sentiment_data()
        X = df['text']
        y = df['sentiment']
        
    print("\n--- Model Training ---")
    
    # 1. Text Vectorization (TF-IDF)
    print("1. Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
    X_vec = vectorizer.fit_transform(X)
    
    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    
    # 3. Model Training (Logistic Regression is great for baseline text classification)
    print("2. Training Logistic Regression Classifier...")
    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # 4. Evaluation
    print("3. Evaluating Model on Test Set...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    # 5. Saving the models
    os.makedirs('ml_pipeline', exist_ok=True)
    
    print("\n4. Saving Models...")
    vec_path = 'ml_pipeline/tfidf_vectorizer.pkl'
    model_path = 'ml_pipeline/sentiment_model.pkl'
    
    with open(vec_path, 'wb') as f:
        pickle.dump(vectorizer, f)
        
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
        
    print(f"✅ Vectorizer successfully saved to {vec_path}")
    print(f"✅ Sentiment Model successfully saved to {model_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        train_sentiment_model(sys.argv[1])
    else:
        train_sentiment_model()
