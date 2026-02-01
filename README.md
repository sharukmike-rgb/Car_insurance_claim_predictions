# 🛡️ Insurance Claim Prediction: End-to-End Machine Learning Project

### **Project by: Michael Sharuk**  
**Objective:** To build a predictive model that identifies high-risk insurance policies and deploy it as an interactive web dashboard.

---

## 📖 1. Project Overview
In the insurance industry, predicting which policyholders are likely to file a claim is critical for financial stability and risk management. This project analyzes historical data to predict the `is_claim` target variable.

### **The Challenge:**
The dataset is highly **imbalanced**, with only ~6% of cases being actual claims. This requires specific machine learning techniques like cost-sensitive learning to avoid a biased model.

---

## 🛠️ 2. Technical Stack
* **Data Analysis:** Pandas, NumPy
* **Visualizations:** Matplotlib, Seaborn, Plotly
* **Machine Learning:** Scikit-Learn, XGBoost
* **Deployment:** Streamlit
* **Serialization:** Joblib

---

## ⚙️ 3. Machine Learning Pipeline
I implemented a robust pipeline to ensure data integrity and prevent data leakage:

1.  **Preprocessing:** * **Numerical Features:** Scaled using `StandardScaler`.
    * **Categorical Features:** Transformed using `OneHotEncoder`.
    * **Pipeline:** Used `ColumnTransformer` to bundle these steps.
2.  **Model Selection:** XGBoost was chosen for its ability to handle complex non-linear relationships and its built-in support for imbalanced data via `scale_pos_weight`.
3.  **Evaluation:** Focused on **ROC-AUC** and **Recall** rather than simple accuracy.



---

## 📊 4. Key Insights from EDA
* **Feature Importance:** Features like `policy_tenure` and `age_of_car` were found to be the top drivers of claim probability.
* **Class Imbalance:** Visualized the sharp contrast between claim and no-claim instances to justify the use of specialized loss functions.

---

## 🚀 5. Deployment & User Interface
The project is deployed via a **Streamlit Dashboard** which features:
* **Real-time Risk Assessment:** Users can input policy details and get an instant probability score.
* **Interactive Charts:** Dynamic visualizations using Plotly to explore historical trends.
* **Template Injection:** A technical workaround to handle the 40+ required model features while keeping the UI simple for the user.



---

## 📂 6. How to Run This Project
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/sharukmike-rgb/Car_insurance_claim_predictions]
    ```
2.  **Install Dependencies:**
    ```bash
    pip install pandas scikit-learn xgboost streamlit joblib plotly
    ```
3.  **Launch the App:**
    ```bash
    streamlit run app.py
    ```

---

## ✅ 7. Conclusion
This project successfully demonstrates the transition from raw data to a deployable AI product. It addresses real-world insurance challenges through careful data engineering and modern machine learning frameworks.
