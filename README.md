# Vendor Invoice Intelligence Portal

An end-to-end Machine Learning application that predicts freight costs and identifies high-risk vendor invoices using historical procurement and invoice data. The project combines data preprocessing, machine learning, model deployment, and an interactive Streamlit dashboard to support financial decision-making.

---

## Project Overview

The Vendor Invoice Intelligence Portal is designed to help finance and procurement teams by:

- Predicting freight costs for vendor invoices.
- Identifying invoices that may require manual approval based on predefined risk patterns.
- Providing an interactive web interface for real-time predictions.

The project consists of two machine learning modules:

1. **Freight Cost Prediction** (Regression)
2. **Invoice Risk Flagging** (Classification)

---

## Dataset

The project uses data stored in an SQLite database (`inventory.db`) containing vendor invoice and purchase information.

### Data Sources

- `vendor_invoice`
- `purchases`

The invoice risk dataset is created by joining these tables using SQL and generating additional aggregated features.

---

## Features

### Freight Cost Prediction

Predicts freight cost using:

- Quantity
- Invoice Dollars

### Invoice Risk Flagging

Predicts whether an invoice should be flagged for manual approval using:

- Invoice Quantity
- Invoice Dollars
- Freight
- Total Item Quantity
- Total Item Dollars

Risk labels are generated based on business rules such as:

- Invoice amount mismatch
- High receiving delay

---

## Project Structure

```text
Vendor Invoice Intelligence Portal
│
├── freight_cost_prediction/
│   ├── data_preprocessing.py
│   ├── model_evaluation.py
│   ├── train.py
│   └── models/
│
├── invoice_flagging/
│   ├── data_preprocessing.py
│   ├── modeling_evaluation.py
│   ├── train.py
│   └── models/
│
├── inference/
│   ├── predict_freight.py
│   └── predict_invoice_flag.py
│
├── app.py
├── inventory.db
└── README.md
```

---

## Technologies Used

### Programming

- Python

### Libraries

- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit

### Database

- SQLite
- SQL

---

## Machine Learning Models

### Freight Cost Prediction

Regression models evaluated:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

The best-performing model is saved for inference.

### Invoice Risk Flagging

Classification model:

- Random Forest Classifier

Hyperparameter tuning is performed using:

- GridSearchCV

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report

---

## Workflow

### Freight Cost Prediction

1. Load vendor invoice data from SQLite.
2. Select features and target.
3. Split training and testing data.
4. Train multiple regression models.
5. Evaluate model performance.
6. Save the best model.
7. Predict freight cost for new invoices.

### Invoice Risk Flagging

1. Load invoice and purchase data.
2. Aggregate purchase information using SQL.
3. Generate invoice risk labels.
4. Scale input features.
5. Train Random Forest Classifier.
6. Tune hyperparameters using GridSearchCV.
7. Save trained model and scaler.
8. Predict invoice risk for new records.

---

## Streamlit Application

The project includes an interactive Streamlit dashboard with two prediction modules.

### Freight Cost Prediction

Users can enter:

- Quantity
- Invoice Dollars

The application predicts the estimated freight cost.

### Invoice Manual Approval Prediction

Users can enter:

- Invoice Quantity
- Invoice Dollars
- Freight
- Total Item Quantity
- Total Item Dollars

The application predicts whether the invoice should be flagged for manual approval.

---

## Model Files

### Regression

- `predict_freight_model.pkl`

### Classification

- `predict_flag_invoice.pkl`

### Feature Scaling

- `scaler.pkl`

---

## How to Run

### Clone the repository

```bash
git clone <repository-url>
cd Vendor-Invoice-Intelligence-Portal
```

### Install dependencies

```bash
pip install pandas numpy scikit-learn streamlit joblib
```

### Train Models

Freight Cost Prediction

```bash
cd freight_cost_prediction
python train.py
```

Invoice Risk Flagging

```bash
cd invoice_flagging
python train.py
```

### Launch the Application

```bash
streamlit run app.py
```

---

## Results

- Built separate machine learning pipelines for regression and classification.
- Automated freight cost prediction using historical invoice data.
- Identified high-risk invoices using engineered business rules and Random Forest classification.
- Deployed both models through an interactive Streamlit interface for real-time predictions.

---

## Skills Demonstrated

- Python Programming
- Machine Learning
- Regression
- Classification
- Feature Engineering
- Data Preprocessing
- SQL
- SQLite
- Model Evaluation
- Hyperparameter Tuning
- GridSearchCV
- Streamlit
- Joblib
- Data Analytics
