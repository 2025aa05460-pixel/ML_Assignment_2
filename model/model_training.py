"""
ML Assignment 2 - Classification Models Training
Student: ARJUN RANA
BITS ID: 2025aa05460
Email: 2025aa05460@wilp.bits-pilani.ac.in

Dataset: Wine Quality Dataset (UCI Machine Learning Repository)
Task: Binary Classification - Predict wine quality (Good: quality >= 7, Bad: quality < 7)
Features: 12 (11 physicochemical properties + wine type)
Instances: 6,497 (1,599 red + 4,898 white)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

# Create directories if they don't exist
os.makedirs('../data', exist_ok=True)

print("=" * 80)
print("ML ASSIGNMENT 2 - CLASSIFICATION MODELS")
print("Student: ARJUN RANA | BITS ID: 2025aa05460")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD AND EXPLORE THE DATASET
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: LOADING AND EXPLORING THE DATASET")
print("=" * 80)

# Load Wine Quality Dataset
print("Loading Wine Quality Dataset...")

# Load red and white wine data
red_wine = pd.read_csv('../data/winequality-red.csv', sep=';')
red_wine['wine_type'] = 0  # Red = 0

white_wine = pd.read_csv('../data/winequality-white.csv', sep=';')
white_wine['wine_type'] = 1  # White = 1

# Combine datasets
df = pd.concat([red_wine, white_wine], ignore_index=True)

print(f"\nDataset Shape: {df.shape}")
print(f"Number of Features: {df.shape[1] - 1}")  # Excluding target
print(f"Number of Instances: {df.shape[0]}")

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Original Quality Distribution ---")
print(df['quality'].value_counts().sort_index())

# ============================================================================
# STEP 2: DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: DATA PREPROCESSING")
print("=" * 80)

# Create a copy for preprocessing
data = df.copy()

# Convert quality to binary classification
# Good wine: quality >= 7, Bad wine: quality < 7
data['target'] = (data['quality'] >= 7).astype(int)

print("\n--- Binary Target Distribution ---")
print(f"Good Wine (quality >= 7): {data['target'].sum()}")
print(f"Bad Wine (quality < 7): {len(data) - data['target'].sum()}")
print(f"\nClass Distribution (%):\n{data['target'].value_counts(normalize=True) * 100}")

# Drop original quality column
data = data.drop('quality', axis=1)

# Check for missing values
print(f"\n--- Missing Values ---")
print(data.isnull().sum())

# Feature columns (excluding target)
feature_cols = [col for col in data.columns if col != 'target']
print(f"\nFeature Columns ({len(feature_cols)}): {feature_cols}")

# ============================================================================
# STEP 3: TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: TRAIN-TEST SPLIT")
print("=" * 80)

# Separate features and target
X = data.drop('target', axis=1)
y = data['target']

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining Set Size: {X_train.shape[0]}")
print(f"Test Set Size: {X_test.shape[0]}")
print(f"Number of Features: {X_train.shape[1]}")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, 'scaler.pkl')

# Save test data for Streamlit app
test_data = X_test.copy()
test_data['target'] = y_test.values
test_data.to_csv('../data/test_data.csv', index=False)
print("\nTest data saved to '../data/test_data.csv'")

# Save feature names
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'feature_names.pkl')

# ============================================================================
# STEP 4: TRAIN ALL CLASSIFICATION MODELS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: TRAINING CLASSIFICATION MODELS")
print("=" * 80)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, 
                             eval_metric='logloss', verbosity=0)
}

# Dictionary to store results
results = {}

# Function to calculate all metrics
def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate all required evaluation metrics."""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_prob),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    return metrics

# Train and evaluate each model
for name, model in models.items():
    print(f"\n--- Training {name} ---")
    
    # Use scaled data for models that benefit from scaling
    if name in ['Logistic Regression', 'K-Nearest Neighbors']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    results[name] = metrics
    
    # Save the trained model
    model_filename = name.lower().replace(' ', '_').replace('-', '_') + '.pkl'
    joblib.dump(model, model_filename)
    print(f"Model saved as: {model_filename}")
    
    # Print metrics
    print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"  AUC:       {metrics['AUC']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall:    {metrics['Recall']:.4f}")
    print(f"  F1 Score:  {metrics['F1']:.4f}")
    print(f"  MCC:       {metrics['MCC']:.4f}")

# ============================================================================
# STEP 5: RESULTS SUMMARY AND COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: RESULTS SUMMARY - MODEL COMPARISON")
print("=" * 80)

# Create results DataFrame
results_df = pd.DataFrame(results).T
results_df = results_df.round(4)
results_df.index.name = 'Model'
results_df = results_df.reset_index()

print("\n" + "-" * 80)
print("COMPARISON TABLE: ALL MODELS AND METRICS")
print("-" * 80)
print(results_df.to_string(index=False))

# Save results to CSV
results_df.to_csv('../data/model_results.csv', index=False)
print("\nResults saved to '../data/model_results.csv'")

# ============================================================================
# STEP 6: VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: CREATING VISUALIZATIONS")
print("=" * 80)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Model Performance Comparison - Wine Quality Classification', fontsize=16, fontweight='bold')

metrics_list = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
colors = sns.color_palette('viridis', n_colors=len(results_df))

for idx, metric in enumerate(metrics_list):
    ax = axes[idx // 3, idx % 3]
    bars = ax.bar(results_df['Model'], results_df[metric], color=colors)
    ax.set_title(metric, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, val in zip(bars, results_df[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('../data/model_comparison.png', dpi=150, bbox_inches='tight')
print("Comparison chart saved to '../data/model_comparison.png'")

# Create confusion matrices for all models
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Confusion Matrices for All Models', fontsize=16, fontweight='bold')

for idx, (name, model) in enumerate(models.items()):
    ax = axes[idx // 3, idx % 3]
    
    # Get predictions
    if name in ['Logistic Regression', 'K-Nearest Neighbors']:
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Bad Wine', 'Good Wine'], yticklabels=['Bad Wine', 'Good Wine'])
    ax.set_title(name, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig('../data/confusion_matrices.png', dpi=150, bbox_inches='tight')
print("Confusion matrices saved to '../data/confusion_matrices.png'")

# ============================================================================
# STEP 7: MODEL OBSERVATIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: MODEL OBSERVATIONS")
print("=" * 80)

observations = {
    'Logistic Regression': """
    - Good baseline performance with decent accuracy on wine quality classification
    - Fast training and inference time, suitable for real-time predictions
    - Works well on this dataset as features show linear relationships with quality
    - AUC score indicates reasonable discrimination ability between wine qualities
    - Interpretable model - coefficients show feature importance for wine quality
    """,
    
    'Decision Tree': """
    - Captures non-linear relationships in wine chemical properties
    - Easily interpretable - can visualize decision rules for wine classification
    - No feature scaling required, handles raw wine measurements directly
    - May show lower generalization due to dataset imbalance (fewer good wines)
    - Tree depth limited to prevent overfitting on training wine samples
    """,
    
    'K-Nearest Neighbors': """
    - Performance depends heavily on k value and feature scaling
    - Scaled features (StandardScaler) help compare wine properties properly
    - Computationally expensive but effective for wine similarity-based classification
    - Works well when similar wines have similar quality ratings
    - Sensitive to outliers in wine chemical measurements
    """,
    
    'Naive Bayes': """
    - Assumes feature independence which may not hold for correlated wine properties
    - Very fast training and prediction, good for quick baseline
    - May underperform as wine chemical properties are often correlated
    - Handles the class imbalance reasonably well
    - Simple probabilistic model provides calibrated probability estimates
    """,
    
    'Random Forest': """
    - Strong performance through ensemble of decision trees on wine data
    - Reduces overfitting by averaging multiple tree predictions
    - Provides feature importance showing key wine quality indicators
    - Robust to outliers and noise in wine measurements
    - Handles class imbalance better than single decision tree
    """,
    
    'XGBoost': """
    - Often achieves best performance with gradient boosting optimization
    - Regularization helps prevent overfitting on imbalanced wine data
    - Handles the class imbalance (more bad wines than good) effectively
    - Feature importance reveals most predictive wine characteristics
    - Efficient implementation suitable for production deployment
    """
}

for model_name, observation in observations.items():
    print(f"\n{model_name}:")
    print(observation)

# ============================================================================
# STEP 8: SAVE FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: FINAL SUMMARY")
print("=" * 80)

# Find best model for each metric
print("\n--- Best Model per Metric ---")
for metric in metrics_list:
    best_model = results_df.loc[results_df[metric].idxmax(), 'Model']
    best_value = results_df[metric].max()
    print(f"{metric}: {best_model} ({best_value:.4f})")

# Overall best model (based on F1 score - balanced metric)
best_overall = results_df.loc[results_df['F1'].idxmax(), 'Model']
print(f"\n*** Best Overall Model (by F1 Score): {best_overall} ***")

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print("\nFiles saved:")
print("  - Models: logistic_regression.pkl, decision_tree.pkl, k_nearest_neighbors.pkl,")
print("            naive_bayes.pkl, random_forest.pkl, xgboost.pkl")
print("  - Scaler: scaler.pkl")
print("  - Feature Names: feature_names.pkl")
print("  - Test Data: ../data/test_data.csv")
print("  - Results: ../data/model_results.csv")
print("  - Visualizations: ../data/model_comparison.png, ../data/confusion_matrices.png")
print("\n" + "=" * 80)
