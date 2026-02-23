"""
================================================================================
AUTOMATED ML PIPELINE TEMPLATE (WITH METRIC SELECTION)
================================================================================

This template automates the entire ML pipeline after preprocessing.
You only need to:
1. Complete the PREPROCESSING SECTION (Steps 1-7)
2. Run the script

Everything else (model selection, tuning, evaluation, plotting) runs automatically.

================================================================================
"""

# =============================================================================
# IMPORTS (DO NOT MODIFY)
# =============================================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Sklearn - Preprocessing
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Sklearn - Model Selection & Evaluation
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# Sklearn - Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Sklearn - Metrics
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

# Plotting
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


# =============================================================================
# PREPROCESSING SECTION - COMPLETE THIS PART FOR EACH NEW DATASET
# =============================================================================

"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: LOAD THE DATASET                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ Options:                                                                    │
│   - pd.read_csv("path/to/file.csv")                                        │
│   - pd.read_csv("https://url/to/data.csv")                                 │
│   - sklearn.datasets.load_XXX()                                            │
│                                                                             │
│ Check:                                                                      │
│   - df.shape → (rows, columns)                                             │
│   - df.head() → preview first rows                                         │
│   - df.info() → column types and non-null counts                           │
│   - df.describe() → statistics for numerical columns                       │
└─────────────────────────────────────────────────────────────────────────────┘
"""

# YOUR CODE HERE:
# df = pd.read_csv("your_data.csv")
# OR
# from sklearn.datasets import load_breast_cancer
# data = load_breast_cancer()
# df = pd.DataFrame(data.data, columns=data.feature_names)
# df['target'] = data.target

df = None  # ← REPLACE WITH YOUR DATAFRAME


"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: IDENTIFY AND SET THE TARGET COLUMN NAME                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ The target is what you want to predict (e.g., 'Survived', 'target', 'y')   │
│                                                                             │
│ Check target distribution:                                                  │
│   - df[TARGET].value_counts() → class balance                              │
│   - Imbalanced? Consider stratify=y in train_test_split                    │
└─────────────────────────────────────────────────────────────────────────────┘
"""

# YOUR CODE HERE:
TARGET = 'target'  # ← REPLACE WITH YOUR TARGET COLUMN NAME


"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: DROP NON-RELEVANT COLUMNS                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ Remove columns that are:                                                    │
│   - IDs (PassengerId, CustomerID, etc.)                                    │
│   - Too many missing values (>50%)                                         │
│   - Irrelevant (Name, Ticket numbers, etc.)                                │
│   - Redundant (highly correlated with other features)                      │
│                                                                             │
│ Check:                                                                      │
│   - df.isna().mean() * 100 → percentage missing per column                 │
│   - df.nunique() → number of unique values (high = possibly ID)            │
└─────────────────────────────────────────────────────────────────────────────┘
"""

# YOUR CODE HERE:
COLUMNS_TO_DROP = []  # ← ADD COLUMN NAMES TO DROP, e.g., ['PassengerId', 'Name']


"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: CHECK FOR MISSING VALUES                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ Run:                                                                        │
│   - df.isna().sum() → count missing per column                             │
│   - df.isna().mean() * 100 → percentage missing                            │
│                                                                             │
│ Decide imputation strategy:                                                 │
│   - Numerical: 'mean' (normal distribution) or 'median' (skewed/outliers)  │
│   - Categorical: 'most_frequent' (mode)                                    │
│   - High missing (>50%): Consider dropping the column                      │
└─────────────────────────────────────────────────────────────────────────────┘
"""

# Check missing values (run this to see what you're dealing with)
# print(df.isna().sum())
# print(df.isna().mean() * 100)


"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: IDENTIFY FEATURE TYPES                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ Separate features into:                                                     │
│   - NUMERICAL: Continuous values (age, price, temperature)                 │
│   - CATEGORICAL: Discrete categories (gender, color, country)              │
│                                                                             │
│ Auto-detect:                                                                │
│   - df.select_dtypes(include=['object']).columns → categorical             │
│   - df.select_dtypes(include=['number']).columns → numerical               │
│                                                                             │
│ WARNING: Some integers are categorical (e.g., Pclass = 1,2,3)              │
│          Check with df[col].nunique() - if few unique values, it's likely  │
│          categorical even if numeric dtype.                                │
└─────────────────────────────────────────────────────────────────────────────┘
"""

# YOUR CODE HERE:
numerical_features = []    # ← ADD NUMERICAL COLUMN NAMES, e.g., ['Age', 'Fare']
categorical_features = []  # ← ADD CATEGORICAL COLUMN NAMES, e.g., ['Sex', 'Embarked']


"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: BUILD THE PREPROCESSOR                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ Common patterns:                                                            │
│                                                                             │
│ A) ALL NUMERICAL (no missing values):                                      │
│    preprocessor = Pipeline([('scaler', StandardScaler())])                 │
│                                                                             │
│ B) ALL NUMERICAL (with missing values):                                    │
│    preprocessor = Pipeline([                                               │
│        ('imputer', SimpleImputer(strategy='median')),                      │
│        ('scaler', StandardScaler())                                        │
│    ])                                                                       │
│                                                                             │
│ C) MIXED NUMERICAL + CATEGORICAL:                                          │
│    numeric_transformer = Pipeline([                                        │
│        ('imputer', SimpleImputer(strategy='median')),                      │
│        ('scaler', StandardScaler())                                        │
│    ])                                                                       │
│    categorical_transformer = Pipeline([                                    │
│        ('imputer', SimpleImputer(strategy='most_frequent')),               │
│        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))│
│    ])                                                                       │
│    preprocessor = ColumnTransformer([                                      │
│        ('num', numeric_transformer, numerical_features),                   │
│        ('cat', categorical_transformer, categorical_features)              │
│    ])                                                                       │
│                                                                             │
│ D) ALL NUMERICAL (no preprocessing needed - rare):                         │
│    preprocessor = 'passthrough'                                            │
└─────────────────────────────────────────────────────────────────────────────┘
"""

# YOUR CODE HERE - UNCOMMENT AND MODIFY ONE OF THE PATTERNS ABOVE:

# Pattern A: All numerical, no missing
# preprocessor = Pipeline([('scaler', StandardScaler())])

# Pattern B: All numerical, with missing
# preprocessor = Pipeline([
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())
# ])

# Pattern C: Mixed numerical + categorical
# numeric_transformer = Pipeline([
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())
# ])
# categorical_transformer = Pipeline([
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
# ])
# preprocessor = ColumnTransformer([
#     ('num', numeric_transformer, numerical_features),
#     ('cat', categorical_transformer, categorical_features)
# ])

preprocessor = None  # ← REPLACE WITH YOUR PREPROCESSOR


"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 7: CHOOSE YOUR OPTIMIZATION METRIC                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ Choose based on your problem:                                               │
│                                                                             │
│   'accuracy'  → Balanced classes, all errors equally bad                   │
│   'precision' → False positives are costly (spam filter, stock picks)      │
│   'recall'    → False negatives are costly (disease detection, fraud)      │
│   'f1'        → Need balance between precision and recall                  │
│   'roc_auc'   → Imbalanced data, care about ranking/probability            │
│                                                                             │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ DECISION GUIDE:                                                         │ │
│ │                                                                         │ │
│ │ Q: "What's worse - missing a positive or a false alarm?"                │ │
│ │                                                                         │ │
│ │    → Missing a positive is worse    = RECALL                            │ │
│ │    → False alarm is worse           = PRECISION                         │ │
│ │    → Both equally bad               = F1 or ACCURACY                    │ │
│ │    → Data is imbalanced             = ROC_AUC or F1                     │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│ EXAMPLES:                                                                   │
│ ┌──────────────────────┬─────────────┬────────────────────────────────────┐ │
│ │ Problem              │ Metric      │ Reasoning                          │ │
│ ├──────────────────────┼─────────────┼────────────────────────────────────┤ │
│ │ Cancer detection     │ recall      │ Don't miss cancer cases!           │ │
│ │ Spam filter          │ precision   │ Don't lose important emails!       │ │
│ │ Fraud detection      │ recall / f1 │ Catch fraud, tolerate some alerts  │ │
│ │ Stock investment     │ precision   │ Only invest when confident         │ │
│ │ Customer churn       │ f1 / roc_auc│ Balance retention costs            │ │
│ │ Balanced dataset     │ accuracy    │ Simple and interpretable           │ │
│ │ Imbalanced dataset   │ roc_auc / f1│ Accuracy would be misleading       │ │
│ └──────────────────────┴─────────────┴────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
"""

# YOUR CODE HERE:
SCORING_METRIC = 'accuracy'  # ← CHOOSE: 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'

# Optional: Brief justification (helps you remember why you chose this metric)
METRIC_JUSTIFICATION = ""  # e.g., "Medical diagnosis - missing cancer is worse than false alarm"


# =============================================================================
# AUTOMATED SECTION - DO NOT MODIFY BELOW THIS LINE
# =============================================================================

def run_ml_pipeline(df, target_col, columns_to_drop, preprocessor, scoring_metric):
    """
    Runs the complete ML pipeline automatically.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset
    target_col : str
        Name of the target column
    columns_to_drop : list
        Columns to remove before modeling
    preprocessor : sklearn transformer
        The preprocessing pipeline/transformer
    scoring_metric : str
        Metric to optimize: 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'
    
    Returns:
    --------
    dict : Results containing best model, metrics, and predictions
    """
    
    print("=" * 70)
    print("AUTOMATED ML PIPELINE")
    print("=" * 70)
    print(f"\n🎯 Optimization Metric: {scoring_metric.upper()}")
    
    # Metric descriptions for user clarity
    metric_descriptions = {
        'accuracy': 'Overall correctness (best for balanced data)',
        'precision': 'When I predict positive, am I right? (minimize false positives)',
        'recall': 'Did I find all positives? (minimize false negatives)',
        'f1': 'Balance between precision and recall',
        'roc_auc': 'Ranking ability across all thresholds'
    }
    print(f"   → {metric_descriptions.get(scoring_metric, 'Custom metric')}")
    
    # Metric function mapping
    metric_functions = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
        'roc_auc': roc_auc_score
    }
    
    # =========================================================================
    # STEP 8: PREPARE DATA
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 8: DATA PREPARATION")
    print("=" * 70)
    
    # Drop specified columns
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop, errors='ignore')
        print(f"\nDropped columns: {columns_to_drop}")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target array shape: {y.shape}")
    print(f"\nTarget distribution:\n{y.value_counts()}")
    print(f"\nTarget percentage:\n{(y.value_counts(normalize=True) * 100).round(2)}")
    
    # Check for class imbalance
    class_ratio = y.value_counts(normalize=True).min()
    if class_ratio < 0.3:
        print(f"\n⚠️  WARNING: Class imbalance detected (minority class = {class_ratio*100:.1f}%)")
        if scoring_metric == 'accuracy':
            print("   Consider using 'f1', 'roc_auc', or 'recall' instead of 'accuracy'")
    
    # =========================================================================
    # STEP 9: TRAIN-TEST SPLIT
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 9: TRAIN-TEST SPLIT")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.0f}%)")
    print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.0f}%)")
    print(f"\nClass distribution in y_train:\n{y_train.value_counts(normalize=True).round(4)}")
    print(f"\nClass distribution in y_test:\n{y_test.value_counts(normalize=True).round(4)}")
    
    # =========================================================================
    # STEP 10: MODEL SELECTION
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"STEP 10: MODEL SELECTION & COMPARISON (optimizing {scoring_metric})")
    print("=" * 70)
    
    # Define all models with preprocessor
    models = {
        'LogisticRegression': Pipeline([
            ('preprocessor', preprocessor),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'DecisionTree': Pipeline([
            ('preprocessor', preprocessor),
            ('clf', DecisionTreeClassifier(random_state=42))
        ]),
        'RandomForest': Pipeline([
            ('preprocessor', preprocessor),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        'GradientBoost': Pipeline([
            ('preprocessor', preprocessor),
            ('clf', GradientBoostingClassifier(random_state=42))
        ]),
        'XGBoost': Pipeline([
            ('preprocessor', preprocessor),
            ('clf', XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                eval_metric='logloss',
                random_state=42
            ))
        ])
    }
    
    # Train and evaluate each model
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate score based on chosen metric
        if scoring_metric == 'roc_auc':
            y_proba = model.predict_proba(X_test)[:, 1]
            score = roc_auc_score(y_test, y_proba)
        else:
            score = metric_functions[scoring_metric](y_test, y_pred)
        
        results[name] = score
        print(f"  {scoring_metric.capitalize()}: {score:.4f}")
    
    # Find best model
    best_model_name = max(results, key=results.get)
    print(f"\n{'*' * 50}")
    print(f"*** BEST MODEL by {scoring_metric.upper()}: {best_model_name} ({results[best_model_name]:.4f}) ***")
    print(f"{'*' * 50}")
    
    # =========================================================================
    # STEP 11: HYPERPARAMETER TUNING (GridSearchCV)
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"STEP 11: HYPERPARAMETER TUNING (optimizing {scoring_metric})")
    print("=" * 70)
    
    # Comprehensive parameter grids for all models
    param_grids = {
        'LogisticRegression': {
            'clf__C': [0.01, 0.1, 1, 10],
            'clf__penalty': ['l1', 'l2'],
            'clf__solver': ['liblinear']
        },
        'DecisionTree': {
            'clf__max_depth': [5, 10, 20, None],
            'clf__min_samples_split': [2, 5, 10],
            'clf__min_samples_leaf': [1, 2, 4],
            'clf__criterion': ['gini', 'entropy']
        },
        'RandomForest': {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [10, 20, None],
            'clf__min_samples_split': [2, 5],
            'clf__min_samples_leaf': [1, 2],
            'clf__max_features': ['sqrt', 'log2']
        },
        'GradientBoost': {
            'clf__n_estimators': [100, 200],
            'clf__learning_rate': [0.05, 0.1, 0.2],
            'clf__max_depth': [3, 5, 7],
            'clf__subsample': [0.8, 1.0]
        },
        'XGBoost': {
            'clf__n_estimators': [100, 200],
            'clf__learning_rate': [0.05, 0.1, 0.2],
            'clf__max_depth': [3, 5, 7],
            'clf__subsample': [0.8, 1.0]
        }
    }
    
    best_model_pipe = models[best_model_name]
    param_grid = param_grids[best_model_name]
    
    n_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"\nTuning {best_model_name}...")
    print(f"Parameter combinations to try: {n_combinations}")
    print(f"Total fits (combinations × 5 folds): {n_combinations * 5}")
    
    grid_search = GridSearchCV(
        estimator=best_model_pipe,
        param_grid=param_grid,
        cv=5,
        scoring=scoring_metric,  # ← Uses the chosen metric!
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest CV {scoring_metric.capitalize()}: {grid_search.best_score_:.4f}")
    print(f"Best Parameters: {grid_search.best_params_}")
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # =========================================================================
    # STEP 12: FINAL EVALUATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 12: FINAL EVALUATION ON TEST SET")
    print("=" * 70)
    
    # Predictions
    y_pred_test = best_model.predict(X_test)
    y_proba_test = best_model.predict_proba(X_test)[:, 1]
    
    # Calculate ALL metrics (regardless of which one was optimized)
    accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    roc_auc = roc_auc_score(y_test, y_proba_test)
    
    # Store metrics in dict for easy access
    all_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }
    
    print(f"\n{'=' * 40}")
    print("FINAL TEST SET RESULTS")
    print(f"{'=' * 40}")
    
    # Print all metrics, highlighting the optimized one
    for metric_name, metric_value in all_metrics.items():
        indicator = " ← OPTIMIZED" if metric_name == scoring_metric else ""
        print(f"{metric_name.capitalize():10s}: {metric_value:.4f}{indicator}")
    
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred_test)}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test)
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Neg   Pos")
    print(f"Actual Neg   {cm[0,0]:4d}  {cm[0,1]:4d}  (TN, FP)")
    print(f"       Pos   {cm[1,0]:4d}  {cm[1,1]:4d}  (FN, TP)")
    
    # Interpretation help
    print(f"\nInterpretation:")
    print(f"  True Negatives (TN):  {cm[0,0]:4d} - Correctly predicted negative")
    print(f"  False Positives (FP): {cm[0,1]:4d} - Incorrectly predicted positive (Type I error)")
    print(f"  False Negatives (FN): {cm[1,0]:4d} - Incorrectly predicted negative (Type II error)")
    print(f"  True Positives (TP):  {cm[1,1]:4d} - Correctly predicted positive")
    
    # =========================================================================
    # STEP 13: CROSS-VALIDATION REPORT
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 13: CROSS-VALIDATION STABILITY CHECK")
    print("=" * 70)
    
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring=scoring_metric)
    
    print(f"\n5-Fold CV Scores ({scoring_metric}): {cv_scores.round(4)}")
    print(f"Mean CV {scoring_metric.capitalize()}: {cv_scores.mean():.4f}")
    print(f"Std CV {scoring_metric.capitalize()}:  {cv_scores.std():.4f}")
    print(f"95% CI: [{cv_scores.mean() - 1.96*cv_scores.std():.4f}, {cv_scores.mean() + 1.96*cv_scores.std():.4f}]")
    
    # Overfitting check (using accuracy for consistency)
    train_acc = best_model.score(X_train, y_train)
    test_acc = accuracy
    print(f"\nOverfitting Check (using accuracy):")
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy:  {test_acc:.4f}")
    print(f"  Difference:     {train_acc - test_acc:.4f}")
    
    if train_acc - test_acc > 0.1:
        print("  ⚠️  WARNING: Possible overfitting (>10% gap)")
    elif train_acc - test_acc < -0.05:
        print("  ⚠️  WARNING: Possible underfitting or data leakage")
    else:
        print("  ✅ Model appears well-fitted")
    
    # =========================================================================
    # STEP 14: PLOTTING
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 14: GENERATING PLOTS")
    print("=" * 70)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Add main title with metric info
    fig.suptitle(f'ML Pipeline Results (Optimized for {scoring_metric.upper()})', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    # 1. Confusion Matrix
    ConfusionMatrixDisplay.from_estimator(
        best_model, X_test, y_test, 
        ax=axes[0, 0], cmap='Blues'
    )
    axes[0, 0].set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    
    # 2. ROC Curve
    RocCurveDisplay.from_estimator(best_model, X_test, y_test, ax=axes[0, 1])
    axes[0, 1].set_title(f'ROC Curve (AUC = {roc_auc:.4f})', fontsize=12, fontweight='bold')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    
    # 3. Cross-Validation Scores
    fold_numbers = range(1, 6)
    bars = axes[1, 0].bar(fold_numbers, cv_scores, color='steelblue', edgecolor='black')
    axes[1, 0].axhline(y=cv_scores.mean(), color='red', linestyle='--', 
                       label=f'Mean: {cv_scores.mean():.4f}')
    axes[1, 0].set_xlabel('Fold')
    axes[1, 0].set_ylabel(scoring_metric.capitalize())
    axes[1, 0].set_title(f'5-Fold Cross-Validation ({scoring_metric.capitalize()})', 
                         fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].set_ylim(min(cv_scores) - 0.05, 1.0)
    
    # 4. Model Comparison
    model_names = list(results.keys())
    scores = list(results.values())
    colors = ['green' if name == best_model_name else 'steelblue' for name in model_names]
    
    bars = axes[1, 1].barh(model_names, scores, color=colors, edgecolor='black')
    axes[1, 1].set_xlabel(scoring_metric.capitalize())
    axes[1, 1].set_title(f'Model Comparison ({scoring_metric.capitalize()})', 
                         fontsize=12, fontweight='bold')
    axes[1, 1].set_xlim(min(scores) - 0.05, 1.0)
    
    # Add value labels
    for bar, score in zip(bars, scores):
        axes[1, 1].text(score + 0.005, bar.get_y() + bar.get_height()/2, 
                        f'{score:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('ml_pipeline_results.png', dpi=150, bbox_inches='tight')
    print("\n✅ Plots saved to 'ml_pipeline_results.png'")
    plt.show()
    
    # =========================================================================
    # STEP 15: FEATURE IMPORTANCE (if applicable)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 15: FEATURE IMPORTANCE")
    print("=" * 70)
    
    try:
        clf = best_model.named_steps['clf']
        
        if hasattr(clf, 'feature_importances_'):
            # Tree-based models
            importances = clf.feature_importances_
            feature_names = X.columns if hasattr(X, 'columns') else [f'Feature {i}' for i in range(len(importances))]
            
        elif hasattr(clf, 'coef_'):
            # Linear models
            importances = np.abs(clf.coef_[0])
            feature_names = X.columns if hasattr(X, 'columns') else [f'Feature {i}' for i in range(len(importances))]
        else:
            importances = None
            
        if importances is not None:
            # Get top 10 features
            importance_df = pd.DataFrame({
                'feature': feature_names[:len(importances)],
                'importance': importances
            }).sort_values('importance', ascending=True).tail(10)
            
            plt.figure(figsize=(10, 6))
            plt.barh(importance_df['feature'], importance_df['importance'], color='steelblue')
            plt.xlabel('Importance')
            plt.title(f'Top 10 Feature Importances ({best_model_name})', fontweight='bold')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
            print("\n✅ Feature importance plot saved to 'feature_importance.png'")
            plt.show()
            
            print("\nTop 10 Most Important Features:")
            for _, row in importance_df.iloc[::-1].iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
    except Exception as e:
        print(f"\n⚠️  Could not extract feature importances: {e}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"""
    Configuration:
      - Dataset:           {X.shape[0]} samples, {X.shape[1]} features
      - Optimization:      {scoring_metric.upper()}
      - Best Model:        {best_model_name}
      - Best Parameters:   {grid_search.best_params_}
    
    Performance (on test set):
      - Accuracy:          {accuracy:.4f}
      - Precision:         {precision:.4f}
      - Recall:            {recall:.4f}
      - F1-Score:          {f1:.4f}
      - ROC-AUC:           {roc_auc:.4f}
      
    Cross-Validation ({scoring_metric}):
      - Mean CV Score:     {cv_scores.mean():.4f} ± {cv_scores.std():.4f}
      - 95% CI:            [{cv_scores.mean() - 1.96*cv_scores.std():.4f}, {cv_scores.mean() + 1.96*cv_scores.std():.4f}]
    
    Files Generated:
      - ml_pipeline_results.png (main plots)
      - feature_importance.png (if applicable)
    """)
    
    return {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'best_params': grid_search.best_params_,
        'model_comparison': results,
        'metrics': all_metrics,
        'optimized_metric': scoring_metric,
        'cv_scores': cv_scores,
        'predictions': y_pred_test,
        'probabilities': y_proba_test,
        'confusion_matrix': cm
    }


# =============================================================================
# RUN THE PIPELINE
# =============================================================================

if __name__ == "__main__":
    
    # Validate preprocessing configuration
    if df is None:
        print("❌ ERROR: Please load your dataset into 'df'")
        print("   Example: df = pd.read_csv('your_data.csv')")
    elif preprocessor is None:
        print("❌ ERROR: Please define your 'preprocessor'")
        print("   See Step 6 for examples")
    else:
        # Run the pipeline
        results = run_ml_pipeline(
            df=df,
            target_col=TARGET,
            columns_to_drop=COLUMNS_TO_DROP,
            preprocessor=preprocessor,
            scoring_metric=SCORING_METRIC  # ← Uses your chosen metric!
        )
        
        print("\n✅ Pipeline completed successfully!")
        print("   Access results with: results['best_model'], results['metrics'], etc.")