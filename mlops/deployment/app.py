import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

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
        font-size: 1.1rem;
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
st.markdown('<div class="sub-header">AI-Powered Predictive Risk Analytics & Relationship Management Tool</div>', unsafe_allow_html=True)

# Sidebar - System Info & Presets
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/bank.png", width=70)
    st.title("Control Panel")
    st.markdown("---")
    
    st.subheader("💡 Load Scenario Presets")
    preset = st.radio(
        "Choose a sample profile:",
        ["Custom Input", "⚠️ High Churn Risk Profile", "✅ Low Churn Risk Profile"]
    )
    
    # Set Preset Values
    if preset == "⚠️ High Churn Risk Profile":
        def_credit = 590
        def_geo = "Germany"
        def_age = 52
        def_tenure = 2
        def_balance = 125000.0
        def_num_prod = 1
        def_card = "Yes"
        def_active = "No"
        def_salary = 75000.0
    elif preset == "✅ Low Churn Risk Profile":
        def_credit = 750
        def_geo = "France"
        def_age = 28
        def_tenure = 7
        def_balance = 45000.0
        def_num_prod = 2
        def_card = "Yes"
        def_active = "Yes"
        def_salary = 95000.0
    else:
        def_credit = 650
        def_geo = "France"
        def_age = 38
        def_tenure = 5
        def_balance = 50000.0
        def_num_prod = 1
        def_card = "Yes"
        def_active = "Yes"
        def_salary = 60000.0

    st.markdown("---")
    st.subheader("⚙️ Model Metadata")
    st.info("""
    - **Model Architecture**: Tuned XGBoost Pipeline
    - **Decision Threshold**: `0.45`
    - **Host**: Hugging Face Hub
    - **Repo**: `krish21may/Bank-Customer-Churn-4`
    """)

# Main Content Layout
col_input, col_results = st.columns([1.1, 1], gap="large")

with col_input:
    st.subheader("📋 Customer Attributes Input")
    st.markdown("Provide customer demographics and banking relationship parameters below:")
    
    with st.container():
        st.markdown('##### 👤 Demographics & Tenure')
        c1, c2 = st.columns(2)
        with c1:
            Age = st.number_input("Age (Years)", min_value=18, max_value=100, value=def_age, step=1)
            Geography = st.selectbox("Geography", ["France", "Germany", "Spain"], index=["France", "Germany", "Spain"].index(def_geo))
        with c2:
            Tenure = st.number_input("Tenure (Years with Bank)", min_value=0, max_value=20, value=def_tenure, step=1)
            EstimatedSalary = st.number_input("Estimated Annual Salary ($)", min_value=0.0, max_value=500000.0, value=float(def_salary), step=1000.0)

        st.markdown('##### 💳 Financial Profile & Banking Products')
        c3, c4 = st.columns(2)
        with c3:
            CreditScore = st.number_input("Credit Score", min_value=300, max_value=900, value=def_credit, step=5)
            Balance = st.number_input("Account Balance ($)", min_value=0.0, max_value=1000000.0, value=float(def_balance), step=5000.0)
        with c4:
            NumOfProducts = st.slider("Number of Bank Products", min_value=1, max_value=4, value=def_num_prod)
            HasCrCard = st.selectbox("Has Credit Card?", ["Yes", "No"], index=0 if def_card == "Yes" else 1)
            IsActiveMember = st.selectbox("Is Active Member?", ["Yes", "No"], index=0 if def_active == "Yes" else 1)

    predict_btn = st.button("🔍 Analyze Customer Churn Risk")

# DataFrame Construction
input_data = pd.DataFrame([{
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
    st.subheader("📊 Intelligence & Risk Report")
    
    if predict_btn or preset != "Custom Input":
        if model is not None:
            # Predict Probability & Class
            prediction_proba = float(model.predict_proba(input_data)[0, 1])
            classification_threshold = 0.45
            is_churn = prediction_proba >= classification_threshold
            
            # Risk Gauge Progress Bar
            st.markdown("##### 🎯 Churn Risk Score")
            st.progress(prediction_proba)
            
            # Key Metric Display
            m1, m2 = st.columns(2)
            with m1:
                st.metric(
                    label="Predicted Risk Probability", 
                    value=f"{prediction_proba * 100:.1f}%",
                    delta=f"{(prediction_proba - classification_threshold) * 100:+.1f}% vs threshold",
                    delta_color="inverse"
                )
            with m2:
                if is_churn:
                    st.metric(label="Risk Status", value="HIGH CHURN RISK", delta="Action Required", delta_color="inverse")
                else:
                    st.metric(label="Risk Status", value="LOW CHURN RISK", delta="Stable Customer", delta_color="normal")
            
            # Detailed Alert Banner
            if is_churn:
                st.error(f"""
                ### ⚠️ High Risk of Customer Churn Detected
                The predictive model indicates a **{prediction_proba * 100:.1f}%** probability that this customer will exit the bank (Classification Threshold: 45.0%).
                """)
                
                st.markdown("#### 💡 Recommended Retention Strategies:")
                st.markdown("""
                - 📞 **Relationship Outreach**: Assign a dedicated personal relationship manager for proactive check-in.
                - 💰 **Loyalty & Rate Incentives**: Offer premium savings interest rates or fee waivers on high-balance accounts.
                - 🚀 **Product Engagement Drive**: Introduce complementary financial products aligned with the customer's demographic profile.
                """)
            else:
                st.success(f"""
                ### ✅ Low Churn Risk / Retained Customer
                The predictive model indicates a **{prediction_proba * 100:.1f}%** probability of churn. Customer engagement is healthy.
                """)
                
                st.markdown("#### 📈 Recommended Upsell Opportunities:")
                st.markdown("""
                - 💳 **Credit Card Upgrade**: Offer premium rewards or cash-back credit card options.
                - 📈 **Wealth Management**: Invite customer to explore investment and retirement portfolio services.
                """)
        else:
            st.warning("Model could not be loaded. Please verify Hugging Face hub credentials or network connectivity.")
    else:
        st.info("👈 Fill in the customer details on the left and click **'Analyze Customer Churn Risk'** to view the real-time AI risk assessment report.")

# Payload Inspection Tab
with st.expander("🔍 View Raw Feature Payload"):
    st.dataframe(input_data, use_container_width=True)
