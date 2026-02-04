"""
ML Assignment 2 - Streamlit Web Application
Student: ARJUN RANA
BITS ID: 2025aa05460
Email: 2025aa05460@wilp.bits-pilani.ac.in

Wine Quality Classification - Interactive Demo
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
import os

# Page configuration
st.set_page_config(
    page_title="Wine Quality Classifier - ML Assignment 2",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #722F37;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stDownloadButton {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🍷 Wine Quality Classification</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">ML Assignment 2 | ARJUN RANA | BITS ID: 2025aa05460</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📋 Navigation")
st.sidebar.markdown("---")

# Model selection
st.sidebar.subheader("🤖 Select Model")
model_options = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Decision Tree': 'decision_tree.pkl',
    'K-Nearest Neighbors': 'k_nearest_neighbors.pkl',
    'Naive Bayes': 'naive_bayes.pkl',
    'Random Forest': 'random_forest.pkl',
    'XGBoost': 'xgboost.pkl'
}

selected_model_name = st.sidebar.selectbox(
    "Choose a classification model:",
    list(model_options.keys()),
    index=4  # Default to Random Forest
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 About the Dataset")
st.sidebar.info("""
**Wine Quality Dataset**
- Source: UCI ML Repository
- Features: 12
- Instances: 6,497
- Task: Binary Classification
- Classes: Good Wine (≥7) / Bad Wine (<7)
""")

st.sidebar.markdown("---")
st.sidebar.subheader("👨‍🎓 Student Details")
st.sidebar.text("Name: ARJUN RANA")
st.sidebar.text("BITS ID: 2025aa05460")
st.sidebar.text("Email: 2025aa05460@wilp.bits-pilani.ac.in")


# Load model and scaler
@st.cache_resource
def load_model(model_name):
    """Load the selected model."""
    model_path = os.path.join('model', model_options[model_name])
    return joblib.load(model_path)

@st.cache_resource
def load_scaler():
    """Load the feature scaler."""
    return joblib.load('model/scaler.pkl')

@st.cache_resource
def load_feature_names():
    """Load feature names."""
    return joblib.load('model/feature_names.pkl')

@st.cache_data
def load_model_results():
    """Load pre-computed model results."""
    return pd.read_csv('data/model_results.csv')


# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Predict", "📊 Model Metrics", "🔍 Model Comparison", "ℹ️ About"])

# Tab 1: Upload and Predict
with tab1:
    st.header("Upload Test Data and Get Predictions")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📂 Upload CSV File")
        
        # Download sample test data
        st.markdown("**Don't have test data? Download sample data:**")
        try:
            sample_data = pd.read_csv('data/test_data.csv')
            csv_data = sample_data.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Sample Test Data",
                data=csv_data,
                file_name="sample_test_data.csv",
                mime="text/csv",
                help="Download sample wine quality test data"
            )
        except:
            st.warning("Sample test data not available.")
        
        st.markdown("---")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload your test data (CSV format)",
            type=['csv'],
            help="Upload a CSV file with wine quality features"
        )
    
    with col2:
        st.subheader("📋 Required Features")
        st.markdown("""
        Your CSV should contain these columns:
        1. fixed acidity
        2. volatile acidity
        3. citric acid
        4. residual sugar
        5. chlorides
        6. free sulfur dioxide
        7. total sulfur dioxide
        8. density
        9. pH
        10. sulphates
        11. alcohol
        12. wine_type (0=Red, 1=White)
        
        Optional: `target` column for evaluation
        """)
    
    if uploaded_file is not None:
        st.markdown("---")
        st.subheader("📊 Data Preview & Predictions")
        
        try:
            # Load uploaded data
            df = pd.read_csv(uploaded_file)
            
            # Show data preview
            st.write("**Uploaded Data Preview:**")
            st.dataframe(df.head(10), use_container_width=True)
            st.write(f"Total rows: {len(df)}, Columns: {len(df.columns)}")
            
            # Load model and make predictions
            model = load_model(selected_model_name)
            scaler = load_scaler()
            feature_names = load_feature_names()
            
            # Check if all required features are present
            missing_features = [f for f in feature_names if f not in df.columns]
            
            if missing_features:
                st.error(f"Missing required features: {missing_features}")
            else:
                # Prepare features
                X = df[feature_names]
                
                # Check for target column
                has_target = 'target' in df.columns
                if has_target:
                    y_true = df['target']
                
                # Scale features for certain models
                if selected_model_name in ['Logistic Regression', 'K-Nearest Neighbors']:
                    X_processed = scaler.transform(X)
                else:
                    X_processed = X.values
                
                # Make predictions
                y_pred = model.predict(X_processed)
                y_prob = model.predict_proba(X_processed)[:, 1]
                
                # Add predictions to dataframe
                results_df = df.copy()
                results_df['Predicted'] = y_pred
                results_df['Probability (Good Wine)'] = y_prob.round(4)
                results_df['Prediction Label'] = results_df['Predicted'].map({0: 'Bad Wine', 1: 'Good Wine'})
                
                st.markdown("---")
                st.subheader("🎯 Prediction Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Predictions", len(y_pred))
                with col2:
                    st.metric("Good Wine Predicted", sum(y_pred))
                
                # Show results
                st.write("**Predictions:**")
                display_cols = ['Prediction Label', 'Probability (Good Wine)']
                if has_target:
                    display_cols = ['target', 'Predicted'] + display_cols
                st.dataframe(results_df[display_cols].head(20), use_container_width=True)
                
                # Download predictions
                csv_results = results_df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download All Predictions",
                    data=csv_results,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
                
                # If target is available, show evaluation metrics
                if has_target:
                    st.markdown("---")
                    st.subheader("📈 Model Evaluation on Uploaded Data")
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_true, y_pred)
                    try:
                        auc = roc_auc_score(y_true, y_prob)
                    except:
                        auc = 0.0
                    precision = precision_score(y_true, y_pred, zero_division=0)
                    recall = recall_score(y_true, y_pred, zero_division=0)
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    mcc = matthews_corrcoef(y_true, y_pred)
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.4f}")
                        st.metric("Precision", f"{precision:.4f}")
                    with col2:
                        st.metric("AUC Score", f"{auc:.4f}")
                        st.metric("Recall", f"{recall:.4f}")
                    with col3:
                        st.metric("F1 Score", f"{f1:.4f}")
                        st.metric("MCC", f"{mcc:.4f}")
                    
                    # Confusion Matrix
                    st.markdown("---")
                    st.subheader("🔢 Confusion Matrix")
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    cm = confusion_matrix(y_true, y_pred)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                xticklabels=['Bad Wine', 'Good Wine'],
                                yticklabels=['Bad Wine', 'Good Wine'])
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title(f'Confusion Matrix - {selected_model_name}')
                    st.pyplot(fig)
                    
                    # Classification Report
                    st.subheader("📋 Classification Report")
                    report = classification_report(y_true, y_pred, target_names=['Bad Wine', 'Good Wine'])
                    st.code(report)
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Tab 2: Model Metrics
with tab2:
    st.header(f"📊 {selected_model_name} - Performance Metrics")
    
    try:
        results_df = load_model_results()
        model_metrics = results_df[results_df['Model'] == selected_model_name].iloc[0]
        
        st.markdown("---")
        
        # Display metrics in cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{model_metrics['Accuracy']:.4f}")
            st.metric("AUC Score", f"{model_metrics['AUC']:.4f}")
        
        with col2:
            st.metric("Precision", f"{model_metrics['Precision']:.4f}")
            st.metric("Recall", f"{model_metrics['Recall']:.4f}")
        
        with col3:
            st.metric("F1 Score", f"{model_metrics['F1']:.4f}")
            st.metric("MCC", f"{model_metrics['MCC']:.4f}")
        
        st.markdown("---")
        
        # Visualization of metrics
        st.subheader("📈 Metrics Visualization")
        
        metrics_values = model_metrics[['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']].values
        metrics_names = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
        
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = sns.color_palette('viridis', n_colors=6)
        bars = ax.bar(metrics_names, metrics_values, color=colors)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score')
        ax.set_title(f'{selected_model_name} - Performance Metrics')
        
        # Add value labels
        for bar, val in zip(bars, metrics_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error loading metrics: {str(e)}")

# Tab 3: Model Comparison
with tab3:
    st.header("🔍 Compare All Models")
    
    try:
        results_df = load_model_results()
        
        st.subheader("📊 Performance Comparison Table")
        st.dataframe(results_df.style.highlight_max(axis=0, subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']), 
                     use_container_width=True)
        
        st.markdown("---")
        
        # Comparison charts
        st.subheader("📈 Visual Comparison")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
        colors = sns.color_palette('viridis', n_colors=len(results_df))
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            bars = ax.bar(results_df['Model'], results_df[metric], color=colors)
            ax.set_title(metric, fontweight='bold', fontsize=12)
            ax.set_ylim(0, 1)
            ax.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=8)
            
            # Highlight best
            max_idx = results_df[metric].idxmax()
            bars[max_idx].set_color('#722F37')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Best model summary
        st.subheader("🏆 Best Model per Metric")
        
        col1, col2, col3 = st.columns(3)
        
        for i, metric in enumerate(metrics):
            best_model = results_df.loc[results_df[metric].idxmax(), 'Model']
            best_value = results_df[metric].max()
            
            if i < 2:
                with col1:
                    st.success(f"**{metric}**: {best_model} ({best_value:.4f})")
            elif i < 4:
                with col2:
                    st.success(f"**{metric}**: {best_model} ({best_value:.4f})")
            else:
                with col3:
                    st.success(f"**{metric}**: {best_model} ({best_value:.4f})")
        
    except Exception as e:
        st.error(f"Error loading comparison data: {str(e)}")

# Tab 4: About
with tab4:
    st.header("ℹ️ About This Project")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📚 Dataset Information")
        st.markdown("""
        **Wine Quality Dataset**
        - **Source**: UCI Machine Learning Repository
        - **Total Instances**: 6,497 (1,599 red + 4,898 white)
        - **Features**: 12 physicochemical properties
        - **Target**: Binary classification (Good: quality ≥ 7, Bad: quality < 7)
        
        **Features:**
        1. Fixed acidity
        2. Volatile acidity
        3. Citric acid
        4. Residual sugar
        5. Chlorides
        6. Free sulfur dioxide
        7. Total sulfur dioxide
        8. Density
        9. pH
        10. Sulphates
        11. Alcohol
        12. Wine type (Red=0, White=1)
        """)
    
    with col2:
        st.subheader("🤖 Models Implemented")
        st.markdown("""
        **Classification Models:**
        1. **Logistic Regression** - Linear model for binary classification
        2. **Decision Tree** - Tree-based classifier
        3. **K-Nearest Neighbors** - Instance-based learning
        4. **Naive Bayes** - Probabilistic classifier (Gaussian)
        5. **Random Forest** - Ensemble of decision trees
        6. **XGBoost** - Gradient boosting ensemble
        
        **Evaluation Metrics:**
        - Accuracy
        - AUC (Area Under ROC Curve)
        - Precision
        - Recall
        - F1 Score
        - MCC (Matthews Correlation Coefficient)
        """)
    
    st.markdown("---")
    
    st.subheader("👨‍🎓 Student Information")
    st.markdown("""
    | Field | Value |
    |-------|-------|
    | **Name** | ARJUN RANA |
    | **BITS ID** | 2025aa05460 |
    | **Email** | 2025aa05460@wilp.bits-pilani.ac.in |
    | **Course** | M.Tech (AIML/DSE) - Machine Learning |
    | **Assignment** | Assignment 2 |
    """)
    
    st.markdown("---")
    
    st.subheader("🔗 Links")
    st.markdown("""
    - [GitHub Repository](https://github.com)
    - [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>ML Assignment 2 | Wine Quality Classification | BITS Pilani WILP</p>
        <p>© 2026 ARJUN RANA (2025aa05460)</p>
    </div>
    """,
    unsafe_allow_html=True
)
