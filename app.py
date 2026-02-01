import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# interface
st.set_page_config(page_title="GUVI Capstone: Insurance Predictor", layout="wide", page_icon="🎓")

# data loading & model
@st.cache_resource
def load_assets():
    model = joblib.load('michael_sharuk_model.pkl')
    return model

@st.cache_data
def get_data():
    return pd.read_csv('insurance_dashboard_final.csv')

model = load_assets()
df = get_data()

# navigating in interface
st.sidebar.title("Project Navigation")
page = st.sidebar.radio("Go to:", ["Predictor Tool", "Exploratory Data Analysis", "Project Documentation"])

# pg1- predictor
if page == "Predictor Tool":
    st.title("Insurance Claim Risk Assessment")
    st.markdown("### Interactive ML Interface for Policy holders")
    
    # Top Metrics for Business Context
    m1, m2, m3 = st.columns(3)
    m1.metric("Dataset Size", f"{len(df)} rows")
    m2.metric("Historical Claim Rate", f"{df['is_claim'].mean():.2%}")
    m3.metric("Model Used", "XGBoost")

    st.divider()

    with st.container():
        col_in, col_out = st.columns([1, 1.5])
        
        with col_in:
            st.subheader("📝 Policy Parameters")
            with st.form("user_inputs"):
                # Feature Template (Hidden from user, but satisfies the model's 40-column requirement)
                input_row = df.iloc[0:1].drop('is_claim', axis=1).copy()
                
                # Visible Inputs
                tenure = st.number_input("Policy Tenure (Years)", 0.0, 2.0, 0.5)
                age_car = st.slider("Car Age", 0.0, 1.0, 0.1)
                age_ph = st.slider("Policyholder Age", 18, 100, 35) / 100 # Scaling for model
                cluster = st.selectbox("Area Cluster", df['area_cluster'].unique())
                fuel = st.radio("Fuel Type", df['fuel_type'].unique())
                
                predict_btn = st.form_submit_button("Run Prediction")

        with col_out:
            st.subheader("Prediction Results")
            if predict_btn:
                # Map inputs back to model
                input_row['policy_tenure'] = tenure
                input_row['age_of_car'] = age_car
                input_row['age_of_policyholder'] = age_ph
                input_row['area_cluster'] = cluster
                input_row['fuel_type'] = fuel
                
                # Execution
                prob = model.predict_proba(input_row)[0][1]
                
                if prob > 0.5:
                    st.error(f"## ⚠️ HIGH RISK PROFILE")
                    st.write(f"The model predicts a claim is likely. Probability: **{prob:.2%}**")
                else:
                    st.success(f"## ✅ LOW RISK PROFILE")
                    st.write(f"The model predicts no claim. Probability: **{prob:.2%}**")
                
                st.info("**Model Logic:** This prediction considers factors like engine type and safety features automatically via the template-injection method.")
            else:
                st.write("Adjust the parameters on the left and click 'Run Prediction'.")

# pg-2 - EDA
elif page == "Exploratory Data Analysis":
    st.title(" Data Insights & Trends")
    st.write("Visualizing the features that drive insurance risk.")
    
    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.histogram(df, x="area_cluster", color="is_claim", barmode="group", title="Claims by Area Cluster")
        st.plotly_chart(fig1)
    with c2:
        fig2 = px.box(df, x="is_claim", y="policy_tenure", title="Impact of Tenure on Claims")
        st.plotly_chart(fig2)

# pg-3- documentation
elif page == "Project Documentation":
    st.title("📖 Project Methodology")
    st.markdown("""
    ### Project Overview
    **Developer:** Michael Sharuk  
    **Framework:** GUVI Data Science Capstone  
    
    ### Technical Workflow
    1. **Data Cleaning:** Handled missing values and addressed extreme class imbalance using `scale_pos_weight`.
    2. **Feature Engineering:** Used a `ColumnTransformer` pipeline for simultaneous scaling and encoding.
    3. **Deployment:** Built using Streamlit to provide an end-to-end interface for non-technical users.
    
    ### Performance Metrics
    - **Primary Metric:** ROC-AUC Score (Targeting > 0.70)
    - **Optimization:** Focused on *Recall* to ensure the insurance company doesn't miss potential high-risk payouts.
    """)