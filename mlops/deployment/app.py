import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import joblib
import io
import plotly.express as px
import plotly.graph_objects as go

# Set Page Config
st.set_page_config(
    page_title="Bank Customer Churn Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Theme Selector & System Controls
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/bank.png", width=70)
    st.title("Control Panel")
    
    st.markdown("### 🎨 Theme Selector")
    theme_mode = st.radio(
        "Choose Theme:",
        ["🌙 Dark Mode", "☀️ Light Mode"],
        index=0
    )
    is_dark = "Dark" in theme_mode

    st.markdown("---")
    st.markdown("### ⚙️ System Metadata")
    st.info("""
    - **Model Architecture**: Tuned XGBoost
    - **Decision Threshold**: `0.45`
    - **Hugging Face Hub**: `krish21may/Bank-Customer-Churn-4`
    - **Engine**: Plotly Interactive Visuals
    """)

# Dynamic CSS Theme Tokens
if is_dark:
    bg_color = "#0F172A"
    card_bg = "#1E293B"
    text_primary = "#F8FAFC"
    text_secondary = "#94A3B8"
    border_color = "#334155"
    accent_color = "#38BDF8"
    plotly_template = "plotly_dark"
    card_shadow = "0 4px 12px rgba(0, 0, 0, 0.4)"
else:
    bg_color = "#F8FAFC"
    card_bg = "#FFFFFF"
    text_primary = "#0F172A"
    text_secondary = "#475569"
    border_color = "#E2E8F0"
    accent_color = "#2563EB"
    plotly_template = "plotly_white"
    card_shadow = "0 4px 6px -1px rgba(0, 0, 0, 0.05)"

st.markdown(f"""
<style>
    .stApp {{
        background-color: {bg_color};
        color: {text_primary};
    }}
    .main-header {{
        font-size: 2.2rem;
        font-weight: 800;
        color: {accent_color};
        margin-bottom: 0.2rem;
    }}
    .sub-header {{
        font-size: 1.05rem;
        color: {text_secondary};
        margin-bottom: 1.5rem;
    }}
    .custom-card {{
        background-color: {card_bg};
        border-radius: 12px;
        padding: 20px;
        border: 1px solid {border_color};
        box-shadow: {card_shadow};
        margin-bottom: 20px;
        color: {text_primary};
    }}
    .stButton>button {{
        background-color: {accent_color};
        color: #FFFFFF !important;
        font-size: 1.05rem;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.65rem 2rem;
        width: 100%;
        border: none;
        transition: all 0.3s ease;
    }}
    .stButton>button:hover {{
        opacity: 0.9;
        transform: translateY(-1px);
    }}
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

# Main Title Banner
st.markdown('<div class="main-header">🏦 Bank Customer Churn Intelligence & Visual Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Risk Assessment, Financial Modeling & Dynamic Interactive Dashboards</div>', unsafe_allow_html=True)

# Navigation Tabs
tab_single, tab_batch, tab_analytics = st.tabs([
    "👤 Single Customer Prediction", 
    "📁 Batch CSV Upload & Inference", 
    "📊 Portfolio Visual Analytics"
])

REQUIRED_COLUMNS = [
    'CreditScore', 'Geography', 'Age', 'Tenure', 
    'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
]

# ==============================================================================
# TAB 1: SINGLE CUSTOMER PREDICTION & GAUGE CHART
# ==============================================================================
with tab_single:
    st.markdown("### 👤 Single Customer Risk Profiler")
    
    # Preset Profiles
    c_preset, _ = st.columns([2, 1])
    with c_preset:
        preset = st.radio(
            "Load Sample Customer Scenario:",
            ["Custom Input", "⚠️ High Churn Risk Customer", "✅ Low Churn Risk Customer"],
            horizontal=True
        )
    
    if preset == "⚠️ High Churn Risk Customer":
        def_credit, def_geo, def_age, def_tenure = 590, "Germany", 52, 2
        def_balance, def_num_prod, def_card, def_active, def_salary = 125000.0, 1, "Yes", "No", 75000.0
    elif preset == "✅ Low Churn Risk Customer":
        def_credit, def_geo, def_age, def_tenure = 750, "France", 28, 7
        def_balance, def_num_prod, def_card, def_active, def_salary = 45000.0, 2, "Yes", "Yes", 95000.0
    else:
        def_credit, def_geo, def_age, def_tenure = 650, "France", 38, 5
        def_balance, def_num_prod, def_card, def_active, def_salary = 50000.0, 1, "Yes", "Yes", 60000.0

    col_input, col_results = st.columns([1.1, 1], gap="large")

    with col_input:
        st.markdown("##### 📋 Demographic & Financial Inputs")
        
        c1, c2 = st.columns(2)
        with c1:
            Age = st.number_input("Age (Years)", min_value=18, max_value=100, value=def_age, step=1, key="s_age")
            Geography = st.selectbox("Geography", ["France", "Germany", "Spain"], index=["France", "Germany", "Spain"].index(def_geo), key="s_geo")
            Tenure = st.number_input("Tenure (Years with Bank)", min_value=0, max_value=20, value=def_tenure, step=1, key="s_tenure")
            EstimatedSalary = st.number_input("Estimated Annual Salary ($)", min_value=0.0, max_value=500000.0, value=float(def_salary), step=1000.0, key="s_salary")

        with c2:
            CreditScore = st.number_input("Credit Score", min_value=300, max_value=900, value=def_credit, step=5, key="s_credit")
            Balance = st.number_input("Account Balance ($)", min_value=0.0, max_value=1000000.0, value=float(def_balance), step=5000.0, key="s_balance")
            NumOfProducts = st.slider("Number of Bank Products", min_value=1, max_value=4, value=def_num_prod, key="s_products")
            HasCrCard = st.selectbox("Has Credit Card?", ["Yes", "No"], index=0 if def_card == "Yes" else 1, key="s_card")
            IsActiveMember = st.selectbox("Is Active Member?", ["Yes", "No"], index=0 if def_active == "Yes" else 1, key="s_active")

        predict_btn = st.button("🔍 Predict Customer Churn Risk")

    single_input = pd.DataFrame([{
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
        st.markdown("##### 📊 Real-Time Risk Intelligence")
        
        if predict_btn or preset != "Custom Input":
            if model is not None:
                prob = float(model.predict_proba(single_input)[0, 1])
                threshold = 0.45
                is_churn = prob >= threshold
                
                # Plotly Gauge Chart for Churn Risk
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={'suffix': '%', 'font': {'size': 36, 'color': accent_color}},
                    title={'text': "Churn Risk Index", 'font': {'size': 18, 'color': text_primary}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': text_secondary},
                        'bar': {'color': "#EF4444" if is_churn else "#10B981"},
                        'bgcolor': card_bg,
                        'borderwidth': 1,
                        'bordercolor': border_color,
                        'steps': [
                            {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.2)'},
                            {'range': [30, 45], 'color': 'rgba(245, 158, 11, 0.2)'},
                            {'range': [45, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 3},
                            'thickness': 0.75,
                            'value': threshold * 100
                        }
                    }
                ))
                fig_gauge.update_layout(
                    template=plotly_template,
                    height=250,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Metrics Row
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Predicted Churn Probability", f"{prob * 100:.1f}%", f"{(prob - threshold) * 100:+.1f}% vs threshold", delta_color="inverse")
                with m2:
                    st.metric("Risk Status", "HIGH CHURN RISK" if is_churn else "LOW CHURN RISK", "Action Required" if is_churn else "Stable Customer", delta_color="inverse" if is_churn else "normal")

                if is_churn:
                    st.error(f"### ⚠️ High Risk Customer Flagged ({prob * 100:.1f}%)")
                    st.markdown("""
                    **Recommended Personal Retention Actions:**
                    - 📞 Schedule personal relationship check-in within 48 hours.
                    - 💰 Waive annual fees & offer promotional interest rate on deposits.
                    - 💳 Cross-sell relevant financial protection products.
                    """)
                else:
                    st.success(f"### ✅ Customer Status Stable ({prob * 100:.1f}%)")
                    st.markdown("""
                    **Recommended Engagement Strategy:**
                    - 💳 Offer premium rewards credit card upgrade.
                    - 📈 Invite customer to private wealth advisory seminar.
                    """)
            else:
                st.warning("Model is not loaded properly.")
        else:
            st.info("👈 Enter details on the left and click **'Predict Customer Churn Risk'**.")

# ==============================================================================
# TAB 2: BATCH CSV UPLOAD & BULK INFERENCE
# ==============================================================================
with tab_batch:
    st.markdown("### 📁 Batch Prediction Engine")
    st.markdown("Upload a CSV file of bank customers to generate bulk predictions and automated risk strategies.")
    
    c_up, c_template = st.columns([2, 1], gap="large")
    
    with c_template:
        st.markdown("##### 📥 CSV Template Generator")
        sample_df = pd.DataFrame([
            {'CreditScore': 619, 'Geography': 'France', 'Age': 42, 'Tenure': 2, 'Balance': 0.0, 'NumOfProducts': 1, 'HasCrCard': 1, 'IsActiveMember': 1, 'EstimatedSalary': 101348.88},
            {'CreditScore': 608, 'Geography': 'Spain', 'Age': 41, 'Tenure': 1, 'Balance': 83807.86, 'NumOfProducts': 1, 'HasCrCard': 0, 'IsActiveMember': 1, 'EstimatedSalary': 112542.58},
            {'CreditScore': 502, 'Geography': 'France', 'Age': 42, 'Tenure': 8, 'Balance': 159660.8, 'NumOfProducts': 3, 'HasCrCard': 1, 'IsActiveMember': 0, 'EstimatedSalary': 113931.57},
            {'CreditScore': 699, 'Geography': 'Germany', 'Age': 39, 'Tenure': 1, 'Balance': 120000.0, 'NumOfProducts': 2, 'HasCrCard': 1, 'IsActiveMember': 0, 'EstimatedSalary': 93826.63},
            {'CreditScore': 850, 'Geography': 'Spain', 'Age': 43, 'Tenure': 2, 'Balance': 125510.82, 'NumOfProducts': 1, 'HasCrCard': 1, 'IsActiveMember': 1, 'EstimatedSalary': 79084.1}
        ])
        buf = io.BytesIO()
        sample_df.to_csv(buf, index=False)
        st.download_button(
            "📄 Download Sample CSV Template",
            buf.getvalue(),
            "sample_bank_customers.csv",
            "text/csv"
        )
        
    with c_up:
        uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"File **'{uploaded_file.name}'** loaded ({len(batch_df)} records).")
            
            missing_cols = [c for c in REQUIRED_COLUMNS if c not in batch_df.columns]
            if missing_cols:
                st.error(f"Missing required columns: `{missing_cols}`")
            else:
                proc_df = batch_df[REQUIRED_COLUMNS].copy()
                for b_col in ['HasCrCard', 'IsActiveMember']:
                    if proc_df[b_col].dtype == object:
                        proc_df[b_col] = proc_df[b_col].apply(lambda x: 1 if str(x).strip().lower() in ['1', 'yes', 'true'] else 0)
                
                if model is not None:
                    probs = model.predict_proba(proc_df)[:, 1]
                    threshold = 0.45
                    preds = (probs >= threshold).astype(int)
                    
                    res_df = batch_df.copy()
                    res_df['Churn_Probability_%'] = np.round(probs * 100, 2)
                    res_df['Risk_Status'] = np.where(preds == 1, 'HIGH RISK ⚠️', 'LOW RISK ✅')
                    res_df['Strategy'] = np.where(preds == 1, 'RM Check-in & Rate Waiver', 'Upsell Wealth/Credit')
                    
                    # Store in session state for analytics tab
                    st.session_state['batch_results'] = res_df
                    
                    # Batch Metrics
                    st.markdown("---")
                    st.markdown("##### 📊 Batch Summary KPI Indicators")
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Total Records", len(res_df))
                    k2.metric("High Risk Count", int(np.sum(preds == 1)), f"{np.mean(preds)*100:.1f}% risk rate", delta_color="inverse")
                    k3.metric("Low Risk Count", int(np.sum(preds == 0)))
                    k4.metric("Avg Churn Prob", f"{np.mean(probs)*100:.1f}%")

                    # Donut Chart for Batch Overview
                    st.markdown("---")
                    c_chart1, c_chart2 = st.columns(2)
                    with c_chart1:
                        st.markdown("##### 🎯 Risk Classification Breakdown")
                        fig_donut = px.pie(
                            res_df, 
                            names='Risk_Status', 
                            title="Portfolio Risk Distribution",
                            hole=0.4,
                            color='Risk_Status',
                            color_discrete_map={'HIGH RISK ⚠️': '#EF4444', 'LOW RISK ✅': '#10B981'},
                            template=plotly_template
                        )
                        fig_donut.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_donut, use_container_width=True)
                    
                    with c_chart2:
                        st.markdown("##### 🌍 Churn Risk by Geography")
                        fig_geo = px.histogram(
                            res_df,
                            x='Geography',
                            color='Risk_Status',
                            barmode='group',
                            title="Geographic Risk Comparison",
                            color_discrete_map={'HIGH RISK ⚠️': '#EF4444', 'LOW RISK ✅': '#10B981'},
                            template=plotly_template
                        )
                        fig_geo.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_geo, use_container_width=True)

                    # Table Display
                    st.markdown("##### 📋 Full Batch Predictions Table")
                    st.dataframe(res_df, use_container_width=True)
                    
                    # Download Predictions
                    out_buf = io.BytesIO()
                    res_df.to_csv(out_buf, index=False)
                    st.download_button(
                        "📥 Download Batch Predictions CSV",
                        out_buf.getvalue(),
                        f"churn_predictions_{uploaded_file.name}",
                        "text/csv"
                    )
        except Exception as err:
            st.error(f"Error parsing CSV: {err}")

# ==============================================================================
# TAB 3: PORTFOLIO VISUAL ANALYTICS (PLOTLY DASHBOARD)
# ==============================================================================
with tab_analytics:
    st.markdown("### 📊 Portfolio Visual Analytics Dashboard")
    st.markdown("Interactive visual discovery platform analyzing feature correlations, age clusters, and balance distributions.")
    
    if 'batch_results' in st.session_state and st.session_state['batch_results'] is not None:
        df_ana = st.session_state['batch_results']
    else:
        # Load sample dataset if no batch uploaded yet
        st.info("💡 Displaying baseline analytics dataset. Upload a batch CSV in Tab 2 to visualize your custom portfolio.")
        df_ana = pd.DataFrame([
            {'CreditScore': 619, 'Geography': 'France', 'Age': 42, 'Tenure': 2, 'Balance': 0.0, 'NumOfProducts': 1, 'HasCrCard': 1, 'IsActiveMember': 1, 'EstimatedSalary': 101348.88, 'Churn_Probability_%': 62.4, 'Risk_Status': 'HIGH RISK ⚠️'},
            {'CreditScore': 608, 'Geography': 'Spain', 'Age': 41, 'Tenure': 1, 'Balance': 83807.86, 'NumOfProducts': 1, 'HasCrCard': 0, 'IsActiveMember': 1, 'EstimatedSalary': 112542.58, 'Churn_Probability_%': 18.2, 'Risk_Status': 'LOW RISK ✅'},
            {'CreditScore': 502, 'Geography': 'France', 'Age': 42, 'Tenure': 8, 'Balance': 159660.8, 'NumOfProducts': 3, 'HasCrCard': 1, 'IsActiveMember': 0, 'EstimatedSalary': 113931.57, 'Churn_Probability_%': 74.8, 'Risk_Status': 'HIGH RISK ⚠️'},
            {'CreditScore': 699, 'Geography': 'Germany', 'Age': 55, 'Tenure': 1, 'Balance': 120000.0, 'NumOfProducts': 2, 'HasCrCard': 1, 'IsActiveMember': 0, 'EstimatedSalary': 93826.63, 'Churn_Probability_%': 81.5, 'Risk_Status': 'HIGH RISK ⚠️'},
            {'CreditScore': 850, 'Geography': 'Spain', 'Age': 29, 'Tenure': 7, 'Balance': 45000.0, 'NumOfProducts': 2, 'HasCrCard': 1, 'IsActiveMember': 1, 'EstimatedSalary': 79084.1, 'Churn_Probability_%': 12.1, 'Risk_Status': 'LOW RISK ✅'},
            {'CreditScore': 720, 'Geography': 'Germany', 'Age': 48, 'Tenure': 4, 'Balance': 110000.0, 'NumOfProducts': 1, 'HasCrCard': 1, 'IsActiveMember': 0, 'EstimatedSalary': 88000.0, 'Churn_Probability_%': 68.3, 'Risk_Status': 'HIGH RISK ⚠️'},
            {'CreditScore': 630, 'Geography': 'France', 'Age': 33, 'Tenure': 6, 'Balance': 30000.0, 'NumOfProducts': 2, 'HasCrCard': 1, 'IsActiveMember': 1, 'EstimatedSalary': 55000.0, 'Churn_Probability_%': 15.4, 'Risk_Status': 'LOW RISK ✅'}
        ])

    ca1, ca2 = st.columns(2)
    
    with ca1:
        st.markdown("##### 📈 Age vs. Account Balance vs. Risk Score")
        fig_scatter = px.scatter(
            df_ana,
            x='Age',
            y='Balance',
            size='Churn_Probability_%' if 'Churn_Probability_%' in df_ana.columns else None,
            color='Risk_Status' if 'Risk_Status' in df_ana.columns else 'Geography',
            hover_data=['CreditScore', 'Geography', 'NumOfProducts'],
            title="Customer Demographics & Account Balance Cluster",
            color_discrete_map={'HIGH RISK ⚠️': '#EF4444', 'LOW RISK ✅': '#10B981'},
            template=plotly_template
        )
        fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_scatter, use_container_width=True)

    with ca2:
        st.markdown("##### 🛍️ Churn Distribution by Product Count")
        fig_prod = px.box(
            df_ana,
            x='NumOfProducts',
            y='Churn_Probability_%' if 'Churn_Probability_%' in df_ana.columns else 'Balance',
            color='Risk_Status' if 'Risk_Status' in df_ana.columns else None,
            title="Churn Probability Spread across Product Holdings",
            color_discrete_map={'HIGH RISK ⚠️': '#EF4444', 'LOW RISK ✅': '#10B981'},
            template=plotly_template
        )
        fig_prod.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_prod, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #9CA3AF;'>Bank Customer Churn Intelligence & Analytics Engine | Streamlit • XGBoost • Plotly • Hugging Face</div>", unsafe_allow_html=True)
