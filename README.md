# 🧠 Day 4 – Credit Card Fraud Detection

### 📂 Project Overview
This project detects fraudulent credit card transactions using real-world anonymized data.  
The goal is to accurately identify fraudulent activity from highly imbalanced datasets using advanced machine learning techniques.

---

## 🎯 Objectives
- Understand and visualize the imbalance in transaction data  
- Apply feature scaling and preprocessing  
- Train and compare multiple classification models:
  - Logistic Regression  
  - Random Forest  
  - XGBoost  
- Handle class imbalance using weighting and SMOTE  
- Evaluate using precision, recall, F1-score, and ROC-AUC  
- Save the final trained model for deployment

---

## 🧩 Dataset
**Source:** [Kaggle – Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- **Rows:** 284,807  
- **Features:** 30 (PCA-transformed for privacy)  
- **Target:**  
  - `Class = 0` → Legitimate Transaction  
  - `Class = 1` → Fraudulent Transaction  
- **Imbalance:** Only ~0.17% of all transactions are fraudulent.

---

## ⚙️ Tech Stack
- **Language:** Python  
- **Libraries:** pandas, numpy, scikit-learn, imbalanced-learn, seaborn, matplotlib, xgboost, joblib  
- **Tools:** Jupyter Notebook, GitHub  

---

## 📊 Exploratory Data Analysis
- Verified dataset shape and column details  
- Visualized severe class imbalance  
- Explored correlations of PCA components with target `Class`  
- Scaled `Time` and `Amount` columns using `StandardScaler`

---

## 🧮 Model Training & Evaluation

| Model | Precision (Fraud) | Recall (Fraud) | F1 | ROC-AUC | Notes |
|--------|------------------|----------------|----|----------|-------|
| Logistic Regression | 0.06 | 0.92 | 0.11 | 0.97 | High recall but many false positives |
| Random Forest | 0.96 | 0.75 | 0.85 | 0.96 | Balanced, strong generalization |
| **XGBoost** | **0.85** | **0.84** | **0.85** | **0.98** | ✅ Best trade-off between recall and precision |

**Key Insights:**
- Logistic Regression performed well in ranking (high AUC) but failed at precision.  
- Ensemble models (Random Forest, XGBoost) provided robust performance.  
- XGBoost achieved **ROC-AUC = 0.976**, catching most frauds with minimal false positives.

---

## 🧠 Feature Importance (XGBoost)
Top features contributing to fraud detection:
```
V14, V17, V10, V12, V4, V11, Amount
```
These represent principal components correlated with unusual transaction behavior.

---

## 🧾 Folder Structure
```
Day-4-CreditCard-Fraud/
├── data/
│   └── creditcard.csv
├── notebooks/
│   └── fraud_detection.ipynb
├── models/
│   └── fraud_detector_xgb.joblib
├── requirements.txt
└── README.md
```

---

## 💾 Model Saving
The final trained model is stored as:
```
models/fraud_detector_xgb.joblib
```
It can be easily loaded for deployment:
```python
import joblib
model = joblib.load('models/fraud_detector_xgb.joblib')
```

---

## 🚀 Next Steps
- Integrate into a **FastAPI microservice** for real-time fraud detection  
- Build a **Streamlit dashboard** to visualize predictions  
- Deploy model to **AWS Lambda or Raspberry Pi** for edge inference  

---

## 🧑‍💻 Author
**Ahmed Ali**  
📍 Colchester, United Kingdom  
🔗 [LinkedIn](https://www.linkedin.com/in/ahmed-ali2006/)  
🔗 [GitHub](https://github.com/AhmedAli2006)

---
