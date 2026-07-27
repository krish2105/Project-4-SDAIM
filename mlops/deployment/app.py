import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import joblib
import io
import plotly.express as px
import plotly.graph_objects as go

import sys
import os

# Ensure current directory and parent directory are in Python module search path
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))

try:
    from mlops.analytics.shap_explainer import calculate_shap_contributions
    from mlops.analytics.roi_calculator import calculate_clv, calculate_expected_retention_roi, optimize_decision_threshold
    from mlops.monitoring.drift_monitor import run_drift_analysis
except ModuleNotFoundError:
    try:
        from analytics.shap_explainer import calculate_shap_contributions
        from analytics.roi_calculator import calculate_clv, calculate_expected_retention_roi, optimize_decision_threshold
        from monitoring.drift_monitor import run_drift_analysis
    except Exception as import_err:
        st.error(f"Module import notice: {import_err}")

# Set Page Config
st.set_page_config(
    page_title="Bank Customer Churn Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Theme Selector & System Controls
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/bank.png", width=70)
    st.title("Control Panel")
    
    st.markdown("### 🎨 Visual Theme")
    theme_mode = st.radio(
        "Select Theme:",
        ["🌙 Dark Mode", "☀️ Light Mode"],
        index=0
    )
    is_dark = "Dark" in theme_mode

    st.markdown("---")
    st.markdown("### ⚙️ MLOps Architecture")
    st.info("""
    - **Model**: Tuned XGBoost Classifier
    - **Explainability**: SHAP Attribution
    - **Financials**: CLV & ROI Optimizer
    - **Drift Engine**: Evidently AI Observability
    - **REST API**: FastAPI Microservice
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

# Header Banner
st.markdown('<div class="main-header">🏦 Bank Customer Churn Intelligence & MLOps Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Enterprise Decision System: SHAP Explainability • Financial CLV Optimizer • Evidently AI Drift Observatory</div>', unsafe_allow_html=True)

# Navigation Tabs
tab_single, tab_batch, tab_analytics, tab_drift, tab_api = st.tabs([
    "👤 Single Risk & SHAP XAI", 
    "📁 Batch CSV Processor", 
    "📊 Visual Portfolio Analytics",
    "📉 Evidently Drift Monitor",
    "⚡ FastAPI Microservice"
])

REQUIRED_COLUMNS = [
    'CreditScore', 'Geography', 'Age', 'Tenure', 
    'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
]

# ==============================================================================
# TAB 1: SINGLE CUSTOMER PREDICTION & SHAP EXPLAINABILITY & CLV ROI
# ==============================================================================
with tab_single:
    st.markdown("### 👤 Single Customer Analysis & SHAP Explainability")
    
    c_preset, _ = st.columns([2, 1])
    with c_preset:
        preset = st.radio(
            "Load Scenario Profile:",
            ["Custom Input", "⚠️ High Churn Risk Profile", "✅ Low Churn Risk Profile"],
            horizontal=True
        )
    
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
        st.markdown("##### 📋 Input Demographics & Financial Parameters")
        
        c1, c2 = st.columns(2)
        with c1:
            Age = st.number_input("Age (Years)", min_value=18, max_value=100, value=def_age, step=1, key="s_age")
            Geography = st.selectbox("Geography", ["France", "Germany", "Spain"], index=["France", "Germany", "Spain"].index(def_geo), key="s_geo")
            Tenure = st.number_input("Tenure (Years)", min_value=0, max_value=20, value=def_tenure, step=1, key="s_tenure")
            EstimatedSalary = st.number_input("Estimated Salary ($)", min_value=0.0, max_value=500000.0, value=float(def_salary), step=1000.0, key="s_salary")

        with c2:
            CreditScore = st.number_input("Credit Score", min_value=300, max_value=900, value=def_credit, step=5, key="s_credit")
            Balance = st.number_input("Account Balance ($)", min_value=0.0, max_value=1000000.0, value=float(def_balance), step=5000.0, key="s_balance")
            NumOfProducts = st.slider("Number of Products", min_value=1, max_value=4, value=def_num_prod, key="s_products")
            HasCrCard = st.selectbox("Has Credit Card?", ["Yes", "No"], index=0 if def_card == "Yes" else 1, key="s_card")
            IsActiveMember = st.selectbox("Is Active Member?", ["Yes", "No"], index=0 if def_active == "Yes" else 1, key="s_active")

        predict_btn = st.button("🔍 Run Churn Risk & SHAP Analysis")

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
        st.markdown("##### 📊 Prediction & SHAP Risk Attribution")
        
        if predict_btn or preset != "Custom Input":
            if model is not None:
                prob = float(model.predict_proba(single_input)[0, 1])
                threshold = 0.45
                is_churn = prob >= threshold
                
                # Gauge Chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={'suffix': '%', 'font': {'size': 36, 'color': accent_color}},
                    title={'text': "Churn Probability Index", 'font': {'size': 18, 'color': text_primary}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#EF4444" if is_churn else "#10B981"},
                        'bgcolor': card_bg,
                        'steps': [
                            {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.2)'},
                            {'range': [30, 45], 'color': 'rgba(245, 158, 11, 0.2)'},
                            {'range': [45, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
                        ],
                        'threshold': {'line': {'color': "red", 'width': 3}, 'value': threshold * 100}
                    }
                ))
                fig_gauge.update_layout(template=plotly_template, height=220, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Financial CLV & Retention ROI
                clv = calculate_clv(Balance, EstimatedSalary, NumOfProducts, Tenure)
                roi_info = calculate_expected_retention_roi(prob, clv)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Est. Customer Lifetime Value", f"${clv:,.2f}")
                m2.metric("Expected Saved Value", f"${roi_info['Expected_Saved_Value']:,.2f}")
                m3.metric("Retention Net Impact", f"${roi_info['Net_Financial_Impact']:,.2f}", delta=f"{roi_info['Campaign_ROI_%']}% ROI", delta_color="normal" if roi_info['Net_Financial_Impact']>0 else "inverse")

                # SHAP Feature Drivers Bar Chart
                st.markdown("##### 🔍 Local SHAP Feature Risk Attribution")
                shap_df = calculate_shap_contributions(model, single_input)
                
                fig_shap = px.bar(
                    shap_df,
                    x='SHAP_Impact',
                    y='Feature',
                    orientation='h',
                    color='Impact_Type',
                    color_discrete_map={'Increases Churn Risk ⚠️': '#EF4444', 'Decreases Churn Risk ✅': '#10B981'},
                    title="SHAP Feature Force Drivers",
                    template=plotly_template
                )
                fig_shap.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=280)
                st.plotly_chart(fig_shap, use_container_width=True)

            else:
                st.warning("Model not loaded.")
        else:
            st.info("👈 Set customer details on the left and click **'Run Churn Risk & SHAP Analysis'**.")

# ==============================================================================
# TAB 2: BATCH CSV PROCESSOR
# ==============================================================================
with tab_batch:
    st.markdown("### 📁 Batch Prediction Engine")
    st.markdown("Upload a customer CSV file to perform bulk AI churn risk assessment and CLV financial optimization.")
    
    c_up, c_template = st.columns([2, 1], gap="large")
    
    with c_template:
        st.markdown("##### 📥 Sample CSV Template")
        sample_df = pd.DataFrame([
            {'CreditScore': 619, 'Geography': 'France', 'Age': 42, 'Tenure': 2, 'Balance': 0.0, 'NumOfProducts': 1, 'HasCrCard': 1, 'IsActiveMember': 1, 'EstimatedSalary': 101348.88},
            {'CreditScore': 608, 'Geography': 'Spain', 'Age': 41, 'Tenure': 1, 'Balance': 83807.86, 'NumOfProducts': 1, 'HasCrCard': 0, 'IsActiveMember': 1, 'EstimatedSalary': 112542.58},
            {'CreditScore': 502, 'Geography': 'France', 'Age': 42, 'Tenure': 8, 'Balance': 159660.8, 'NumOfProducts': 3, 'HasCrCard': 1, 'IsActiveMember': 0, 'EstimatedSalary': 113931.57},
            {'CreditScore': 699, 'Geography': 'Germany', 'Age': 39, 'Tenure': 1, 'Balance': 120000.0, 'NumOfProducts': 2, 'HasCrCard': 1, 'IsActiveMember': 0, 'EstimatedSalary': 93826.63},
            {'CreditScore': 850, 'Geography': 'Spain', 'Age': 43, 'Tenure': 2, 'Balance': 125510.82, 'NumOfProducts': 1, 'HasCrCard': 1, 'IsActiveMember': 1, 'EstimatedSalary': 79084.1}
        ])
        buf = io.BytesIO()
        sample_df.to_csv(buf, index=False)
        st.download_button("📄 Download Sample CSV Template", buf.getvalue(), "sample_bank_customers.csv", "text/csv")
        
    with c_up:
        uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"], key="batch_uploader")

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
                    res_df['Churn_Probability'] = probs
                    res_df['Churn_Probability_%'] = np.round(probs * 100, 2)
                    res_df['Risk_Status'] = np.where(preds == 1, 'HIGH RISK ⚠️', 'LOW RISK ✅')
                    
                    # Compute CLV for each row
                    clvs = [
                        calculate_clv(r['Balance'], r['EstimatedSalary'], r['NumOfProducts'], r['Tenure']) 
                        for _, r in proc_df.iterrows()
                    ]
                    res_df['CLV_$'] = np.round(clvs, 2)
                    
                    st.session_state['batch_results'] = res_df
                    st.session_state['batch_proc_df'] = proc_df
                    
                    st.markdown("---")
                    st.markdown("##### 📊 Batch Summary & Optimal Threshold Curve")
                    
                    # Threshold Optimization
                    opt_res = optimize_decision_threshold(res_df)
                    
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Total Records", len(res_df))
                    k2.metric("High Risk Count", int(np.sum(preds == 1)), f"{np.mean(preds)*100:.1f}% risk rate", delta_color="inverse")
                    k3.metric("Optimal Risk Threshold", f"{opt_res['Optimal_Threshold']}")
                    k4.metric("Max Projected Net Profit", f"${opt_res['Max_Net_Profit']:,.2f}")

                    st.dataframe(res_df, use_container_width=True)
                    
                    out_buf = io.BytesIO()
                    res_df.to_csv(out_buf, index=False)
                    st.download_button("📥 Download Batch Predictions CSV", out_buf.getvalue(), f"churn_predictions_{uploaded_file.name}", "text/csv")
        except Exception as err:
            st.error(f"Error parsing CSV: {err}")

# ==============================================================================
# TAB 3: PORTFOLIO VISUAL ANALYTICS
# ==============================================================================
with tab_analytics:
    st.markdown("### 📊 Portfolio Visual Analytics Dashboard")
    
    if 'batch_results' in st.session_state and st.session_state['batch_results'] is not None:
        df_ana = st.session_state['batch_results']
    else:
        st.info("💡 Displaying default analytics dataset. Upload a batch CSV in Tab 2 to visualize your custom portfolio.")
        df_ana = pd.DataFrame([
            {'CreditScore': 619, 'Geography': 'France', 'Age': 42, 'Tenure': 2, 'Balance': 0.0, 'NumOfProducts': 1, 'HasCrCard': 1, 'IsActiveMember': 1, 'EstimatedSalary': 101348.88, 'Churn_Probability_%': 62.4, 'Risk_Status': 'HIGH RISK ⚠️'},
            {'CreditScore': 608, 'Geography': 'Spain', 'Age': 41, 'Tenure': 1, 'Balance': 83807.86, 'NumOfProducts': 1, 'HasCrCard': 0, 'IsActiveMember': 1, 'EstimatedSalary': 112542.58, 'Churn_Probability_%': 18.2, 'Risk_Status': 'LOW RISK ✅'},
            {'CreditScore': 502, 'Geography': 'France', 'Age': 42, 'Tenure': 8, 'Balance': 159660.8, 'NumOfProducts': 3, 'HasCrCard': 1, 'IsActiveMember': 0, 'EstimatedSalary': 113931.57, 'Churn_Probability_%': 74.8, 'Risk_Status': 'HIGH RISK ⚠️'},
            {'CreditScore': 699, 'Geography': 'Germany', 'Age': 55, 'Tenure': 1, 'Balance': 120000.0, 'NumOfProducts': 2, 'HasCrCard': 1, 'IsActiveMember': 0, 'EstimatedSalary': 93826.63, 'Churn_Probability_%': 81.5, 'Risk_Status': 'HIGH RISK ⚠️'}
        ])

    ca1, ca2 = st.columns(2)
    with ca1:
        st.markdown("##### 📈 Demographics & Account Balance Cluster")
        fig_scatter = px.scatter(
            df_ana, x='Age', y='Balance', size='Churn_Probability_%' if 'Churn_Probability_%' in df_ana.columns else None,
            color='Risk_Status' if 'Risk_Status' in df_ana.columns else 'Geography',
            template=plotly_template
        )
        fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_scatter, use_container_width=True)

    with ca2:
        st.markdown("##### 🛍️ Product Holdings vs. Churn Distribution")
        fig_box = px.box(
            df_ana, x='NumOfProducts', y='Churn_Probability_%' if 'Churn_Probability_%' in df_ana.columns else 'Balance',
            color='Risk_Status' if 'Risk_Status' in df_ana.columns else None,
            template=plotly_template
        )
        fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_box, use_container_width=True)

# ==============================================================================
# TAB 4: EVIDENTLY AI DATA DRIFT OBSERVATORY
# ==============================================================================
with tab_drift:
    st.markdown("### 📉 Evidently AI Data Drift Observatory")
    st.markdown("Monitors statistical Data Drift between reference baseline training data (`mlops/data/bank_customer_churn.csv`) and live production traffic.")
    
    if 'batch_proc_df' in st.session_state:
        drift_df = st.session_state['batch_proc_df']
    else:
        # Sample shifted dataset to demonstrate drift detection
        drift_df = pd.DataFrame([
            {'CreditScore': 550, 'Geography': 'Germany', 'Age': 60, 'Tenure': 1, 'Balance': 180000.0, 'NumOfProducts': 3, 'HasCrCard': 1, 'IsActiveMember': 0, 'EstimatedSalary': 120000.0},
            {'CreditScore': 510, 'Geography': 'Spain', 'Age': 58, 'Tenure': 2, 'Balance': 195000.0, 'NumOfProducts': 4, 'HasCrCard': 0, 'IsActiveMember': 0, 'EstimatedSalary': 130000.0}
        ])
        
    drift_results = run_drift_analysis(drift_df)
    
    d1, d2, d3 = st.columns(3)
    d1.metric("Monitoring Status", drift_results['Status'])
    d2.metric("Overall Drift Status", "DRIFT DETECTED ⚠️" if drift_results['Drift_Detected'] else "HEALTHY (NO DRIFT) ✅")
    d3.metric("Drifted Features Share", f"{drift_results['Drift_Share_%']}%")

    if drift_results['Drifted_Features']:
        st.warning(f"⚠️ Features experiencing statistical distribution drift (p < 0.05): `{drift_results['Drifted_Features']}`")
    else:
        st.success("✅ Feature distributions match the baseline dataset. Model remains accurate and compliant.")

    if 'Feature_Metrics' in drift_results:
        st.markdown("##### 🔬 Kolmogorov-Smirnov Statistical P-Values")
        p_vals_df = pd.DataFrame([
            {'Feature': feat, 'KS_Statistic': metrics['statistic'], 'P_Value': metrics['p_value'], 'Drift_Status': 'DRIFTED ⚠️' if metrics['drifted'] else 'NORMAL ✅'}
            for feat, metrics in drift_results['Feature_Metrics'].items()
        ])
        st.dataframe(p_vals_df, use_container_width=True)

# ==============================================================================
# TAB 5: FASTAPI MICROSERVICE DOCS
# ==============================================================================
with tab_api:
    st.markdown("### ⚡ Enterprise FastAPI Microservice")
    st.markdown("High-throughput REST API service for integration into core banking IT infrastructure.")
    
    st.code("""
# Start FastAPI Microservice via Terminal:
uvicorn mlops.api.main:app --host 0.0.0.0 --port 8000 --reload
    """, language="bash")
    
    st.markdown("##### 🔌 Available Endpoints:")
    st.markdown("""
    - `GET /`: Health check & API version
    - `GET /v1/health`: Kubernetes readiness & liveness probe
    - `POST /v1/predict`: Single customer inference with SHAP drivers & CLV ROI
    - `POST /v1/predict-batch`: High-throughput JSON array batch inference
    - `POST /v1/drift-check`: Evidently AI data drift evaluation payload
    """)
    
    st.info("Interactive OpenAPI / Swagger Documentation available at `http://localhost:8000/docs` when service is running.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #9CA3AF;'>Enterprise Bank Customer Churn Intelligence & MLOps Platform | XGBoost • SHAP • Evidently AI • FastAPI • Streamlit</div>", unsafe_allow_html=True)
