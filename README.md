# CreditGuard – Credit Card Fraud Detection System

CreditGuard is a machine learning-based credit card fraud detection project designed to identify potentially fraudulent transactions using supervised and anomaly detection techniques.

The project explores multiple machine learning algorithms, compares their performance, optimises selected models and provides prediction capabilities through a FastAPI backend. SHAP is used to explain model predictions.

## Project Status

**Current stage:** Machine Learning Pipeline and FastAPI Backend Completed

**Next stage:** React Frontend Dashboard

## Key Features

- Exploratory Data Analysis (EDA)
- Duplicate transaction investigation
- Group-aware training, validation and test splitting
- Multiple machine learning model experiments
- Hyperparameter optimisation
- Fraud detection using XGBoost
- SHAP-based model explainability
- REST API using FastAPI
- Automated API testing with Pytest

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python 3.12 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Gradient Boosting | XGBoost, LightGBM, CatBoost |
| Data Visualisation | Matplotlib, Seaborn |
| Explainability | SHAP |
| Backend | FastAPI |
| API Server | Uvicorn |
| Testing | Pytest, HTTPX |
| Dependency Management | uv |
| Development Environment | VS Code, WSL2 Ubuntu |

## Dataset

CreditGuard uses the European credit card fraud detection dataset.

- Total transactions: 284,807
- Legitimate transactions: 284,315
- Fraudulent transactions: 492
- Features: 30 input features
- Target: `Class`

The input features include `Time`, `V1`–`V28`, and `Amount`.

The original dataset is preserved unchanged. Raw and processed datasets are excluded from this repository.

### Data Preparation

- Investigated missing values and duplicate transactions.
- Analysed exact duplicate feature groups and their labels.
- Preserved the original dataset.
- Created group-aware training, validation and test partitions.
- Kept identical feature groups within the same partition to avoid duplicate leakage.

### Dataset Splitting

| Partition | Rows | Fraud cases |
|---|---:|---:|
| Training | 182,276 | 314 |
| Validation | 45,569 | 79 |
| Test | 56,962 | 99 |
| Development total | 227,845 | 393 |

The final test partition was kept separate during model development and used for final evaluation.

## Machine Learning Experiments

The project evaluated the following approaches:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Extra Trees
5. XGBoost
6. LightGBM
7. CatBoost
8. Support Vector Machine
9. Isolation Forest
10. Autoencoder
11. Soft Voting
12. Stacking
13. Hybrid Anomaly Detection

Selected models were further optimised using group-aware cross-validation and Precision-Recall AUC.

No resampling technique was applied in the reported experiments.

## Final Model

**Selected algorithm:** Optimised XGBoost

The final model was trained on the combined development partition containing 227,845 transactions.

### Final Test Performance

| Metric | Result |
|---|---:|
| Accuracy | 99.9579% |
| Precision | 91.21% |
| Recall | 83.84% |
| F1-score | 87.37% |
| PR-AUC | 0.853685 |
| ROC-AUC | 0.992596 |
| MCC | 0.874251 |
| Specificity | 99.9859% |

### Confusion Matrix

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 56,855 | 8 |
| Actual Fraud | 16 | 83 |

The model correctly detected 83 of the 99 fraudulent transactions in the held-out test partition, with 8 false positive predictions.

The classification threshold used for final evaluation was 0.5.

## Model Explainability – SHAP

SHAP was integrated to understand how the trained model makes predictions.

The explainability analysis includes:

- Global feature importance
- SHAP summary plot
- Individual transaction waterfall explanation
- Mean absolute SHAP importance

### Top Influential Features

| Rank | Feature | Mean Absolute SHAP |
|---:|---|---:|
| 1 | V14 | 0.142335 |
| 2 | V4 | 0.074209 |
| 3 | V12 | 0.054972 |
| 4 | V10 | 0.050090 |
| 5 | V3 | 0.035662 |

SHAP values explain the model's feature contributions. They do not establish causal relationships.

## FastAPI Backend

CreditGuard provides a REST API using FastAPI.

### Available Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | Check API and model status |
| POST | `/predict` | Predict transaction fraud probability |

The prediction endpoint accepts all 30 transaction features and returns:

- Prediction: `FRAUD` or `LEGITIMATE`
- Fraud probability
- Fraud probability percentage
- Risk category
- Decision threshold
- Model name

### Example Prediction Response

```json
{
  "prediction": "LEGITIMATE",
  "fraud_probability": 0.001,
  "fraud_probability_percentage": 0.1,
  "risk_category": "LOW",
  "threshold": 0.5,
  "model": "Optimised XGBoost"
}
```

The values above are illustrative.

## Project Structure

```text
CreditGuard/
│
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py
│   │
│   └── ml/
│       ├── experiments/
│       ├── create_splits.py
│       ├── create_validation_split.py
│       ├── explore_data.py
│       ├── analyze_duplicates.py
│       ├── compare_experiments.py
│       ├── final_evaluation.py
│       └── explainability.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   └── figures/
│
├── models/
│   ├── final/
│   └── experiments/
│       └── shap/
│
├── tests/
│   └── test_api.py
│
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

## Installation and Setup

### Prerequisites

- Python 3.12
- uv
- Git

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd CreditGuard
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Add the dataset

Download the original credit card fraud dataset and place it at:

```text
data/raw/creditcard.csv
```

The raw dataset is not included in this repository.

### 4. Start the FastAPI server

```bash
uv run uvicorn backend.api.main:app --reload
```

### 5. Open API documentation

Visit:

http://127.0.0.1:8000/docs

## Automated Testing

CreditGuard uses Pytest to verify API functionality.

Run:

```bash
PYTHONPATH=. uv run pytest tests/test_api.py -v
```

### Current Test Results

- Home endpoint: Passed
- Health endpoint: Passed
- Valid prediction: Passed
- Missing feature validation: Passed

**Result: 4 tests passed.**

## Important Notes

- The original dataset is not modified.
- Raw and processed datasets are excluded from Git.
- The final model and SHAP outputs are included in the project repository.
- The final test partition was not used for model optimisation.
- The dataset does not provide customer identifiers or true calendar timestamps. Results should not be interpreted as proof of generalisation to future transactions or other financial institutions.
- The API is a research and development prototype, not a production-certified banking fraud detection service.

## Future Enhancements

- React and Vite frontend dashboard
- Transaction monitoring dashboard
- Interactive fraud prediction interface
- Visual risk analytics
- SHAP explanations integrated into the dashboard
- Improved API security and validation
- Deployment and monitoring

## Author

**CreditGuard Project**

Machine Learning | Fraud Detection | Explainable AI | FastAPI
