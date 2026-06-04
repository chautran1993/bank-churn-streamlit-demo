# Bank Churn Streamlit Demo

Streamlit demo for predicting bank customer churn using a trained machine learning pipeline.

## Model Summary

- Dataset: Vietnam Bank Churn Dataset 2025
- Best pipeline: `P3-SMOTE+Thr`
- Best model: `Logistic Regression`
- Decision threshold: `0.620`
- Final test metrics: Accuracy `0.7856`, Precision churn `0.4404`, Recall churn `0.7065`, F1 churn `0.5426`, ROC-AUC `0.8574`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app loads `model.pkl` and displays churn probability, risk level, recommended action, and input summary.
