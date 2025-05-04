# 🚨 Fraud Detection System (Machine Learning Project)

A complete end-to-end **fraud detection project** using machine learning models with a production-ready **Streamlit web app**. This project is designed to classify credit card transactions as **legitimate or fraudulent**, based on transaction behavior.

---

## 🌐 Live App

> **[Try the Fraud Detection App](https://shikafraud.streamlit.app/)**
> Upload a CSV with transactions and get fraud predictions instantly.

---

## 🔍 Project Overview

This project addresses the critical issue of detecting **fraudulent financial transactions**, which are rare but highly damaging.

## 🛡️ System Usage Context

This system is designed based on a specific dataset structure (e.g., PCA-transformed features V1–V28, Time, Amount) and assumes similar feature generation and scaling pipelines.
It is not intended for direct use across all institutions, as different companies have unique data schemas and transaction monitoring pipelines.
Therefore, this system is best suited for in-house use, educational prototyping, or adaptation by teams whose input data matches the original format.

### Key Features:

* Cleaned and preprocessed real-world credit card dataset
* Balanced extremely imbalanced data using hybrid resampling
* Trained and compared **Random Forest** and **XGBoost** classifiers
* Tuned thresholds to improve recall without compromising precision
* Built a **Streamlit UI** with full prediction and filtered suspicious view
* Designed a **sequential model logic** (Random Forest ➞ XGBoost confirm)
* Ready for deployment & optional AWS SageMaker integration

---

## 📂 Dataset

* Source: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
* Rows: 284,807 transactions
* Features: 30 anonymized PCA components (V1–V28), `Time`, `Amount`
* Target: `Class` (0 = legitimate, 1 = fraud)

---

## 🪤 Machine Learning Pipeline

### 1. Data Preprocessing

* **Scaled** `Amount` and `Time` using `StandardScaler`
* Created `Hour` feature from `Time` to capture transaction time behavior

### 2. Handling Class Imbalance

* Dataset is \~0.17% fraud (492 out of 284,807)
* I used a **hybrid resampling strategy**:

  * ✅ **RandomUnderSampler**: Reduced majority class to \~10,000
  * ✅ **SMOTE**: Oversampled minority class to match
* Avoids overfitting (too much synthetic data) or underfitting (too little real data)

### 3. Model Training & Selection

| Model         | Reason for Use                                  |
| ------------- | ----------------------------------------------- |
| Random Forest | Robust, interpretable, good default choice      |
| XGBoost       | Powerful gradient boosting, good for comparison |

* Tuned both using `RandomizedSearchCV`
* Evaluated using:

  * Precision, Recall, F1-Score, ROC-AUC
  * Confusion Matrix

### 4. Threshold Tuning

* Default 0.5 threshold is **not optimal** for fraud detection
* Evaluated across thresholds 0.0 to 1.0
* Best thresholds:

  * Random Forest: `0.83`
  * XGBoost: `0.90`

---

## 🚀 Sequential Logic (Production Strategy)

```text
If Random Forest >= 0.83:
    If XGBoost >= 0.99:
        -> Auto-Block
    Else:
        -> Flag for Review
Else:
    -> Allow
```

* Ensures high recall but requires **double confirmation** for critical actions

---

## 🎨 Streamlit Web App Features

* Upload CSV file with correct format
* Show predictions for:

  * Random Forest Probability
  * XGBoost Probability
  * Final Decision: `Allow`, `Flag for Review`, `Auto-Block`
* Extra panel shows only **suspicious transactions**
* Download full results as CSV

---

## 🔧 Local Setup

```bash
git clone https://github.com/your-username/fraud-detection-ml.git
cd fraud-detection-ml
pip install -r requirements.txt
streamlit run fraud_detection_app.py
```

---

## 🛌 Optional: AWS SageMaker Integration (Coming Next)

We plan to:

* Host final model on AWS SageMaker endpoint
* Update Streamlit to call prediction API
* Demonstrate real enterprise-grade deployment

---

## 🏆 Results

| Model         | Precision | Recall | F1 Score | ROC-AUC |
| ------------- | --------- | ------ | -------- | ------- |
| Random Forest | 0.79      | 0.86   | 0.82     | 0.98    |
| XGBoost       | 0.76      | 0.85   | 0.80     | 0.97    |

* ✅ Random Forest chosen as **primary model**
* ✅ XGBoost acts as **confirmation backup**

---

## 🎓 Author

**Treva Antony Ogwang**
Student, Data Science & AI — GISMA University
[GitHub](https://github.com/Begge10850) | [LinkedIn](https://www.linkedin.com/in/treva-ogwang-87235626b/)

---

## 🚩 Disclaimer

This is a student portfolio project. The dataset is anonymized, and this app should not be used for real-world financial decisions.

---

## 📚 License

MIT License
