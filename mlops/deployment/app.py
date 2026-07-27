import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import joblib
import io

# Set Page Config for modern layout
st.set_page_config(
    page_title="Bank Customer Churn Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #F8FAFC;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    .metric-box {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        font-size: 1.05rem;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        width: 100%;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        color: white;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Load Model with Caching
@st.cache_resource
def load_churn_model():
    try:
        model_path = hf_hub_download(
            repo_id="krish21may/Bank-Customer-Churn-4", 
            filename="best_churn_model.joblib"
        )
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model from Hugging Face Hub: {e}")
        return None

model = load_churn_model()

# Header Section
st.markdown('<div class="main-header">🏦 Bank Customer Churn Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Predictive Risk Analytics & Relationship Management Platform</div>', unsafe_allow_html=True)

# Sidebar - System Info & Navigation
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/bank.png", width=70)
    st.title("Control Panel")
    st.markdown("---")
    
    st.subheader("⚙️ System Metadata")
    st.info("""
    - **Model Architecture**: Tuned XGBoost Pipeline
    - **Decision Threshold**: `0.45`
    - **Hugging Face Repo**: `krish21may/Bank-Customer-Churn-4`
    - **Batch Processing**: CSV Multi-Record Inference
    """)

# Main Navigation Tabs
tab_single, tab_batch = st.tabs(["👤 Single Customer Prediction", "📁 Batch Customer CSV Upload"])

# Required Feature Columns
REQUIRED_COLUMNS = [
    'CreditScore', 'Geography', 'Age', 'Tenure', 
    'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
]

# ==============================================================================
# TAB 1: SINGLE CUSTOMER PREDICTION
# ==============================================================================
with tab_single:
    st.markdown("### 👤 Single Customer Analysis")
    
    # Preset Selector
    c_preset, _ = st.columns([2, 1])
    with c_preset:
        preset = st.radio(
            "Load Sample Customer Profile:",
            ["Custom Input", "⚠️ High Churn Risk Profile", "✅ Low Churn Risk Profile"],
            horizontal=True
        )
    
    # Set Preset Values
    if preset == "⚠️ High Churn Risk Profile":
        def_credit, def_geo, def_age, def_tenure = 590, "Germany", 52, 2
        def_balance, def_num_prod, def_card, def_active, def_salary = 125000.0, 1, "Yes", "No", 75000.0
    elif preset == "✅ Low Churn Risk Profile":
        def_credit, def_geo, def_age, def_tenure = 750, "France", 28, 7
        def_balance, def_num_prod, def_card, def_active, def_salary = 45000.0, 2, "Yes", "Yes", 95000.0
    else:
        def_credit, def_geo, def_age, def_tenure = 650, "France", 38, 5
        def_balance, def_num_prod, def_card, def_active, def_salary = 50000.0, 1, "Yes", "Yes", 60000.0

    col_input, col_results = st.columns([1.1, 1], gap="large")

    with col_input:
        st.markdown("##### 📋 Input Demographics & Account Details")
        
        c1, c2 = st.columns(2)
        with c1:
            Age = st.number_input("Age (Years)", min_value=18, max_value=100, value=def_age, step=1, key="single_age")
            Geography = st.selectbox("Geography", ["France", "Germany", "Spain"], index=["France", "Germany", "Spain"].index(def_geo), key="single_geo")
            Tenure = st.number_input("Tenure (Years with Bank)", min_value=0, max_value=20, value=def_tenure, step=1, key="single_tenure")
            EstimatedSalary = st.number_input("Estimated Annual Salary ($)", min_value=0.0, max_value=500000.0, value=float(def_salary), step=1000.0, key="single_salary")

        with c2:
            CreditScore = st.number_input("Credit Score", min_value=300, max_value=900, value=def_credit, step=5, key="single_credit")
            Balance = st.number_input("Account Balance ($)", min_value=0.0, max_value=1000000.0, value=float(def_balance), step=5000.0, key="single_balance")
            NumOfProducts = st.slider("Number of Bank Products", min_value=1, max_value=4, value=def_num_prod, key="single_products")
            HasCrCard = st.selectbox("Has Credit Card?", ["Yes", "No"], index=0 if def_card == "Yes" else 1, key="single_card")
            IsActiveMember = st.selectbox("Is Active Member?", ["Yes", "No"], index=0 if def_active == "Yes" else 1, key="single_active")

        predict_btn = st.button("🔍 Analyze Customer Churn Risk")

    single_input_data = pd.DataFrame([{
        'CreditScore': CreditScore,
        'Geography': Geography,
        'Age': Age,
        'Tenure': Tenure,
        'Balance': Balance,
        'NumOfProducts': NumOfProducts,
        'HasCrCard': 1 if HasCrCard == "Yes" else 0,
        'IsActiveMember': 1 if IsActiveMember == "Yes" else 0,
        'EstimatedSalary': EstimatedSalary
    }])

    with col_results:
        st.markdown("##### 📊 Risk Assessment Report")
        
        if predict_btn or preset != "Custom Input":
            if model is not None:
                prediction_proba = float(model.predict_proba(single_input_data)[0, 1])
                classification_threshold = 0.45
                is_churn = prediction_proba >= classification_threshold
                
                st.markdown("###### 🎯 Churn Risk Score Meter")
                st.progress(prediction_proba)
                
                m1, m2 = st.columns(2)
                with m1:
                    st.metric(
                        label="Predicted Risk Probability", 
                        value=f"{prediction_proba * 100:.1f}%",
                        delta=f"{(prediction_proba - classification_threshold) * 100:+.1f}% threshold diff",
                        delta_color="inverse"
                    )
                with m2:
                    if is_churn:
                        st.metric(label="Risk Classification", value="HIGH RISK", delta="Action Required", delta_color="inverse")
                    else:
                        st.metric(label="Risk Classification", value="LOW RISK", delta="Stable Customer", delta_color="normal")
                
                if is_churn:
                    st.error(f"""
                    ### ⚠️ High Risk of Customer Churn Detected
                    The predictive model indicates a **{prediction_proba * 100:.1f}%** probability that this customer will exit the bank.
                    """)
                    st.markdown("""
                    **Recommended Action Plan:**
                    - 📞 Assign dedicated relationship manager for immediate check-in.
                    - 💰 Offer deposit interest bonus or credit card fee waiver.
                    - 🚀 Promote customized digital banking products.
                    """)
                else:
                    st.success(f"""
                    ### ✅ Low Churn Risk / Retained Customer
                    The predictive model indicates a **{prediction_proba * 100:.1f}%** probability of churn. Customer status is healthy.
                    """)
                    st.markdown("""
                    **Recommended Growth Strategy:**
                    - 💳 Offer premium rewards credit card upgrade.
                    - 📈 Invite customer to explore wealth management services.
                    """)
            else:
                st.warning("Model could not be loaded from Hugging Face Hub.")
        else:
            st.info("👈 Modify parameters on the left and click **'Analyze Customer Churn Risk'** to see results.")

# ==============================================================================
# TAB 2: BATCH CUSTOMER CSV UPLOAD
# ==============================================================================
with tab_batch:
    st.markdown("### 📁 Batch Prediction via CSV Upload")
    st.markdown("Upload a customer CSV file to perform bulk AI churn risk assessment across hundreds of customer accounts simultaneously.")
    
    col_up, col_info = st.columns([2, 1], gap="large")
    
    with col_info:
        st.markdown("##### 📥 Sample CSV Template")
        st.write("Download a sample template with valid columns to format your data:")
        
        sample_df = pd.DataFrame([
            {
                'CreditScore': 619, 'Geography': 'France', 'Age': 42, 'Tenure': 2,
                'Balance': 0.0, 'NumOfProducts': 1, 'HasCrCard': 1, 'IsActiveMember': 1, 'EstimatedSalary': 101348.88
            },
            {
                'CreditScore': 608, 'Geography': 'Spain', 'Age': 41, 'Tenure': 1,
                'Balance': 83807.86, 'NumOfProducts': 1, 'HasCrCard': 0, 'IsActiveMember': 1, 'EstimatedSalary': 112542.58
            },
            {
                'CreditScore': 502, 'Geography': 'France', 'Age': 42, 'Tenure': 8,
                'Balance': 159660.8, 'NumOfProducts': 3, 'HasCrCard': 1, 'IsActiveMember': 0, 'EstimatedSalary': 113931.57
            },
            {
                'CreditScore': 699, 'Geography': 'France', 'Age': 39, 'Tenure': 1,
                'Balance': 0.0, 'NumOfProducts': 2, 'HasCrCard': 0, 'IsActiveMember': 0, 'EstimatedSalary': 93826.63
            },
            {
                'CreditScore': 850, 'Geography': 'Spain', 'Age': 43, 'Tenure': 2,
                'Balance': 125510.82, 'NumOfProducts': 1, 'HasCrCard': 1, 'IsActiveMember': 1, 'EstimatedSalary': 79084.1
            }
        ])
        
        buffer = io.BytesIO()
        sample_df.to_csv(buffer, index=False)
        st.download_button(
            label="📄 Download Sample CSV Template",
            data=buffer.getvalue(),
            file_name="sample_bank_customers.csv",
            mime="text/csv"
        )
    
    with col_up:
        st.markdown("##### 📤 Upload Customer Data File")
        uploaded_file = st.file_uploader("Drop your customer CSV file here", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"File **'{uploaded_file.name}'** uploaded successfully! Found **{len(batch_df)}** records.")
            
            # Validate Required Columns
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in batch_df.columns]
            
            if missing_cols:
                st.error(f"❌ Uploaded CSV is missing required columns: `{missing_cols}`")
                st.info(f"Required columns are: `{REQUIRED_COLUMNS}`")
            else:
                # Preprocess Categoricals / Binary Flags if needed
                processed_df = batch_df[REQUIRED_COLUMNS].copy()
                
                # Convert string binary flags to 1/0 if needed
                for binary_col in ['HasCrCard', 'IsActiveMember']:
                    if processed_df[binary_col].dtype == object:
                        processed_df[binary_col] = processed_df[binary_col].apply(
                            lambda x: 1 if str(x).strip().lower() in ['1', 'yes', 'true'] else 0
                        )
                
                if model is not None:
                    # Run Batch Prediction
                    with st.spinner("Processing AI Batch Predictions..."):
                        probabilities = model.predict_proba(processed_df)[:, 1]
                        classification_threshold = 0.45
                        predictions = (probabilities >= classification_threshold).astype(int)
                        
                        # Append Results
                        results_df = batch_df.copy()
                        results_df['Churn_Probability_%'] = np.round(probabilities * 100, 2)
                        results_df['Risk_Status'] = np.where(predictions == 1, 'HIGH RISK ⚠️', 'LOW RISK ✅')
                        results_df['Recommended_Strategy'] = np.where(
                            predictions == 1, 
                            'Proactive RM Check-in & Rate Bonus', 
                            'Cross-sell Wealth/Credit Products'
                        )
                    
                    st.markdown("---")
                    st.markdown("### 📊 Executive Portfolio Churn Summary")
                    
                    total_count = len(results_df)
                    high_risk_count = int(np.sum(predictions == 1))
                    low_risk_count = total_count - high_risk_count
                    high_risk_pct = (high_risk_count / total_count) * 100
                    avg_prob = np.mean(probabilities) * 100
                    
                    k1, k2, k3, k4 = st.columns(4)
                    with k1:
                        st.metric("Total Customers Processed", f"{total_count:,}")
                    with k2:
                        st.metric("High Risk Customers", f"{high_risk_count:,}", delta=f"{high_risk_pct:.1f}% of total", delta_color="inverse")
                    with k3:
                        st.metric("Low Risk / Retained", f"{low_risk_count:,}")
                    with k4:
                        st.metric("Avg Portfolio Churn Risk", f"{avg_prob:.1f}%")
                    
                    st.markdown("---")
                    st.markdown("### 📋 Detailed Batch Prediction Results")
                    
                    # Filtering Options
                    filter_status = st.selectbox(
                        "Filter Results by Risk Status:",
                        ["All Customers", "High Risk Only (>= 45%)", "Low Risk Only (< 45%)"]
                    )
                    
                    if filter_status == "High Risk Only (>= 45%)":
                        filtered_results = results_df[results_df['Risk_Status'] == 'HIGH RISK ⚠️']
                    elif filter_status == "Low Risk Only (< 45%)":
                        filtered_results = results_df[results_df['Risk_Status'] == 'LOW RISK ✅']
                    else:
                        filtered_results = results_df
                        
                    st.dataframe(filtered_results, use_container_width=True)
                    
                    # Export Button
                    res_buffer = io.BytesIO()
                    results_df.to_csv(res_buffer, index=False)
                    
                    st.download_button(
                        label="📥 Download Full Batch Predictions (CSV)",
                        data=res_buffer.getvalue(),
                        file_name=f"churn_predictions_{uploaded_file.name}",
                        mime="text/csv"
                    )
                else:
                    st.error("Model could not be loaded for batch inference.")
        except Exception as e:
            st.error(f"Error processing CSV file: {e}")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #9CA3AF;'>Bank Customer Churn Intelligence Platform | Powered by XGBoost, MLflow & Hugging Face Hub</div>", unsafe_allow_html=True)
