# Car_insurance_claim_predictions
To build a predictive model that determines whether a customer will make a car insurance claim in the next policy period.
# 🛡️ Insurance Claim Prediction Dashboard

An end-to-end Machine Learning project that predicts the probability of insurance claims using **XGBoost** and provides an interactive interface built with **Streamlit**.

## 📊 Project Overview
The goal of this project is to help insurance providers identify high-risk policyholders. By analyzing features like car age, policy tenure, and geographical area clusters, the model predicts whether a claim (`is_claim`) will be filed. 

The dataset is highly imbalanced (only ~6% claims), so the model utilizes **Cost-Sensitive Learning** (via `scale_pos_weight`) to ensure high recall for the minority class.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn, XGBoost
* **Deployment:** Streamlit
* **Serialization:** Joblib

## 📁 Project Structure
* `insurance_analysis.ipynb`: The notebook containing EDA and model training.
* `app.py`: The Streamlit dashboard code.
* `michael_sharuk_model.pkl`: The trained Scikit-Learn pipeline.
* `insurance_dashboard_final.csv`: Cleaned data used for the dashboard.
* `README.md`: Project documentation.

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the required dependencies:
```bash
pip install pandas scikit-learn xgboost seaborn streamlit joblib
