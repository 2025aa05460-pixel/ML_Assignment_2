# ML Assignment 2 - Wine Quality Classification

## Student Details
| Field | Value |
|-------|-------|
| **Name** | ARJUN RANA |
| **BITS ID** | 2025aa05460 |
| **Email** | 2025aa05460@wilp.bits-pilani.ac.in |
| **Course** | M.Tech (AIML/DSE) - Machine Learning |

---

## Problem Statement

The objective of this assignment is to build a **binary classification system** to predict wine quality based on physicochemical properties. Given various chemical measurements of wine samples, the model classifies wines as either **Good Quality** (quality score ≥ 7) or **Bad Quality** (quality score < 7).

This is a practical machine learning problem that demonstrates:
- Data preprocessing and feature engineering
- Implementation of multiple classification algorithms
- Model evaluation using various metrics
- Deployment of ML models via a web application

---

## Dataset Description

### Source
**Wine Quality Dataset** from UCI Machine Learning Repository  
URL: https://archive.ics.uci.edu/ml/datasets/wine+quality

### Dataset Characteristics
| Property | Value |
|----------|-------|
| **Total Instances** | 6,497 |
| **Red Wine Samples** | 1,599 |
| **White Wine Samples** | 4,898 |
| **Number of Features** | 12 |
| **Target Variable** | Binary (Good/Bad Wine) |
| **Class Distribution** | Good Wine: 19.66% (1,277), Bad Wine: 80.34% (5,220) |

### Features Description

| # | Feature | Description | Type |
|---|---------|-------------|------|
| 1 | fixed acidity | Tartaric acid concentration (g/dm³) | Continuous |
| 2 | volatile acidity | Acetic acid concentration (g/dm³) | Continuous |
| 3 | citric acid | Citric acid concentration (g/dm³) | Continuous |
| 4 | residual sugar | Remaining sugar after fermentation (g/dm³) | Continuous |
| 5 | chlorides | Sodium chloride concentration (g/dm³) | Continuous |
| 6 | free sulfur dioxide | Free SO₂ concentration (mg/dm³) | Continuous |
| 7 | total sulfur dioxide | Total SO₂ concentration (mg/dm³) | Continuous |
| 8 | density | Density of wine (g/cm³) | Continuous |
| 9 | pH | pH value (0-14 scale) | Continuous |
| 10 | sulphates | Potassium sulphate concentration (g/dm³) | Continuous |
| 11 | alcohol | Alcohol content (% by volume) | Continuous |
| 12 | wine_type | Type of wine (0=Red, 1=White) | Binary |

### Target Variable
- **Original**: Quality score (3-9)
- **Transformed**: Binary classification
  - **Good Wine (1)**: Quality ≥ 7
  - **Bad Wine (0)**: Quality < 7

---

## Models Used

Six machine learning classification models were implemented and evaluated:

1. **Logistic Regression** - Linear model for binary classification
2. **Decision Tree Classifier** - Tree-based classification algorithm
3. **K-Nearest Neighbors (KNN)** - Instance-based learning algorithm
4. **Naive Bayes (Gaussian)** - Probabilistic classifier based on Bayes' theorem
5. **Random Forest** - Ensemble method using multiple decision trees
6. **XGBoost** - Gradient boosting ensemble method

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.8223 | 0.8048 | 0.6147 | 0.2617 | 0.3671 | 0.3178 |
| Decision Tree | 0.8508 | 0.8019 | 0.6220 | 0.6172 | 0.6196 | 0.5268 |
| K-Nearest Neighbors | 0.8323 | 0.8264 | 0.5922 | 0.4766 | 0.5281 | 0.4314 |
| Naive Bayes | 0.7392 | 0.7494 | 0.3955 | 0.6133 | 0.4809 | 0.3310 |
| Random Forest (Ensemble) | **0.8885** | **0.9125** | **0.8171** | 0.5586 | 0.6636 | **0.6151** |
| XGBoost (Ensemble) | 0.8792 | 0.9021 | 0.7281 | **0.6172** | **0.6681** | 0.5979 |

**Best Model Overall (by F1 Score): XGBoost**

---

## Model Observations

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Good baseline performance with 82.23% accuracy. Fast training and inference, making it suitable for real-time predictions. The model shows reasonable AUC (0.8048) indicating decent discrimination ability. However, low recall (0.2617) suggests difficulty in identifying good wines, likely due to class imbalance. The model's interpretable coefficients reveal feature importance for wine quality prediction. |
| **Decision Tree** | Achieves 85.08% accuracy by capturing non-linear relationships in wine chemical properties. Balanced precision (0.622) and recall (0.6172) result in good F1 score (0.6196). The tree structure provides interpretable decision rules for classification. Max depth limitation (10) helps prevent overfitting. No feature scaling required, handling raw measurements directly. |
| **K-Nearest Neighbors** | Performance (83.23% accuracy) depends heavily on k value and feature scaling. StandardScaler preprocessing helps compare wine properties properly. Shows moderate recall (0.4766) and good AUC (0.8264). Computationally expensive at prediction time but effective for similarity-based classification. Sensitive to outliers in wine chemical measurements. |
| **Naive Bayes** | Lowest accuracy (73.92%) among all models due to violated independence assumption - wine chemical properties are often correlated. Highest recall (0.6133) after ensemble models, catching more good wines. Very fast training and prediction, suitable for baseline comparison. Simple probabilistic model provides calibrated probability estimates. |
| **Random Forest (Ensemble)** | Best accuracy (88.85%) and AUC (0.9125) among all models. Highest precision (0.8171) means fewer false positives. Reduces overfitting through averaging multiple tree predictions. Provides feature importance rankings showing alcohol, volatile acidity, and density as key quality indicators. Robust to outliers and noise in measurements. |
| **XGBoost (Ensemble)** | Second-best overall with highest F1 score (0.6681). Gradient boosting with regularization prevents overfitting. Handles class imbalance (80% bad wines vs 20% good wines) effectively. Tied for best recall (0.6172) with Decision Tree. Feature importance reveals alcohol content as most predictive. Efficient implementation suitable for production deployment. |

---

## Project Structure

```
project-folder/
│
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
├── model/                    # Model files directory
│   ├── model_training.py     # Model training script
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── k_nearest_neighbors.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl            # Feature scaler
│   └── feature_names.pkl     # Feature names list
│
└── data/                     # Data files directory
    ├── test_data.csv         # Test dataset for evaluation
    ├── model_results.csv     # Model performance metrics
    ├── winequality-red.csv   # Original red wine data
    ├── winequality-white.csv # Original white wine data
    ├── model_comparison.png  # Metrics comparison chart
    └── confusion_matrices.png # Confusion matrices visualization
```

---

## How to Run Locally

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <project-folder>
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**
   - The app will automatically open at `http://localhost:8501`

---

## Streamlit App Features

1. **📤 Upload & Predict**: Upload your own CSV test data and get predictions
2. **📊 Model Metrics**: View detailed performance metrics for selected model
3. **🔍 Model Comparison**: Compare all 6 models side by side
4. **ℹ️ About**: Information about dataset, models, and student details

### App Capabilities
- ✅ CSV file upload for test data
- ✅ Model selection dropdown (6 models)
- ✅ Display of all evaluation metrics
- ✅ Confusion matrix visualization
- ✅ Classification report
- ✅ Download predictions
- ✅ Download sample test data

---

## Links

- **GitHub Repository**: [Repository Link]
- **Live Streamlit App**: [Streamlit App Link]
- **Dataset Source**: [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)

---

## Screenshots

*Screenshots of the BITS Virtual Lab execution will be included in the submitted PDF.*

---

## References

1. P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. "Modeling wine preferences by data mining from physicochemical properties." Decision Support Systems, Elsevier, 47(4):547-553, 2009.
2. UCI Machine Learning Repository - Wine Quality Dataset
3. Scikit-learn Documentation
4. XGBoost Documentation
5. Streamlit Documentation

---

## License

This project is submitted as part of ML Assignment 2 for M.Tech (AIML/DSE) at BITS Pilani WILP.

---

**© 2026 ARJUN RANA (2025aa05460) | BITS Pilani WILP**
