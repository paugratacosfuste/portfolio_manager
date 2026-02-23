import zipfile
import tempfile

def train_sentiment_model(csv_path="ml_pipeline/stock_data.csv"):

    print("=" * 70)
    print("STOCK SENTIMENT NLP MODEL TRAINING")
    print("=" * 70)

    df = None

    # ------------------------------------------------------------------
    # 1. CHECK FOR ZIP FILE FIRST
    # ------------------------------------------------------------------

    zip_filename = "stock_market_sentiment.zip"

    if os.path.exists(zip_filename):
        print(f"Found '{zip_filename}'. Extracting...")

        with tempfile.TemporaryDirectory() as tmpdirname:
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(tmpdirname)

            # Find first CSV inside extracted folder
            csv_files = []
            for root, _, files in os.walk(tmpdirname):
                for file in files:
                    if file.endswith(".csv"):
                        csv_files.append(os.path.join(root, file))

            if not csv_files:
                print("No CSV file found inside ZIP. Falling back to dummy data.")
                df = generate_dummy_sentiment_data()
            else:
                csv_file_path = csv_files[0]
                print(f"Loading extracted CSV: {csv_file_path}")
                df = pd.read_csv(csv_file_path)

    # ------------------------------------------------------------------
    # 2. IF NO ZIP, CHECK FOR DIRECT CSV
    # ------------------------------------------------------------------

    elif os.path.exists(csv_path):
        print(f"Loading '{csv_path}'...")
        df = pd.read_csv(csv_path)

    # ------------------------------------------------------------------
    # 3. FALLBACK TO DUMMY
    # ------------------------------------------------------------------

    else:
        df = generate_dummy_sentiment_data()

    # ------------------------------------------------------------------
    # 4. PROCESS DATA
    # ------------------------------------------------------------------

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
        print(f"Error parsing dataset: {e}. Falling back to dummy data.")
        df = generate_dummy_sentiment_data()
        X = df['text']
        y = df['sentiment']

    print("\n--- Model Training ---")

    # Vectorization
    print("1. Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2
    )

    X_vec = vectorizer.fit_transform(X)

    # Stratified split (important for sentiment balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42, stratify=y
    )

    print("2. Training Logistic Regression Classifier...")
    clf = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0,
        class_weight='balanced'
    )

    clf.fit(X_train, y_train)

    print("3. Evaluating Model on Test Set...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    os.makedirs('ml_pipeline', exist_ok=True)

    print("\n4. Saving Models...")
    vec_path = 'ml_pipeline/tfidf_vectorizer.pkl'
    model_path = 'ml_pipeline/sentiment_model.pkl'

    with open(vec_path, 'wb') as f:
        pickle.dump(vectorizer, f)

    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)

    print(f"Vectorizer successfully saved to {vec_path}")
    print(f"Sentiment Model successfully saved to {model_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        train_sentiment_model(sys.argv[1])
    else:
        train_sentiment_model()
