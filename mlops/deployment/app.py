import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import joblib
import io
import sys
import os

# Scikit-Learn 1.6+ compatibility patch for pickled XGBClassifier models
try:
    from sklearn.base import ClassifierMixin
    from sklearn.utils._tags import Tags, TargetTags, ClassifierTags
    ClassifierMixin.__sklearn_tags__ = lambda self: Tags(
        estimator_type='classifier',
        target_tags=TargetTags(required=False),
        transformer_tags=None,
        regressor_tags=None,
        classifier_tags=ClassifierTags()
    )
except Exception:
    pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import plotly.express as px
import plotly.graph_objects as go

try:
    from mlops.analytics.shap_explainer import calculate_shap_contributions
    from mlops.analytics.roi_calculator import calculate_clv, calculate_expected_retention_roi, optimize_decision_threshold
    from mlops.analytics.counterfactual import generate_counterfactual_scenarios
    from mlops.analytics.survival_analysis import predict_survival_timeline
    from mlops.analytics.uplift_modeling import segment_causal_uplift
    from mlops.analytics.llm_outreach import generate_llm_retention_outreach
    from mlops.analytics.fairness_audit import run_fairness_audit
    from mlops.analytics.monte_carlo_sim import run_monte_carlo_simulation
    from mlops.monitoring.drift_monitor import run_drift_analysis
    from mlops.reports.pdf_generator import generate_executive_pdf_report
except ModuleNotFoundError:
    try:
        from analytics.shap_explainer import calculate_shap_contributions
        from analytics.roi_calculator import calculate_clv, calculate_expected_retention_roi, optimize_decision_threshold
        from analytics.counterfactual import generate_counterfactual_scenarios
        from analytics.survival_analysis import predict_survival_timeline
        from analytics.uplift_modeling import segment_causal_uplift
        from analytics.llm_outreach import generate_llm_retention_outreach
        from analytics.fairness_audit import run_fairness_audit
        from analytics.monte_carlo_sim import run_monte_carlo_simulation
        from monitoring.drift_monitor import run_drift_analysis
        from reports.pdf_generator import generate_executive_pdf_report
    except Exception as import_err:
        st.error(f"Module import notice: {import_err}")

st.set_page_config(
    page_title="E-Commerce Customer Churn Intelligence Platform",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/shopping-cart.png", width=70)
    st.title("Control Panel")
    
    st.markdown("### 🎨 Visual Theme")
    theme_mode = st.radio("Select Theme:", ["🌙 Dark Mode", "☀️ Light Mode"], index=0)
    is_dark = "Dark" in theme_mode

    st.markdown("---")
    st.markdown("### ⚙️ System Architecture")
    st.info("""
    - **Model**: Tuned XGBoost Classifier
    - **XAI**: SHAP Local Feature Attribution
    - **Recourse**: DiCE Counterfactual AI
    - **Survival**: Cox Hazard 24M Curves
    - **Causal**: Uplift Segmentation
    - **Fairness**: Disparate Impact Audit
    - **Domain**: E-Commerce & Retail Subscriptions
    """)

if is_dark:
    bg_color, card_bg, text_primary, text_secondary = "#0F172A", "#1E293B", "#F8FAFC", "#94A3B8"
    border_color, accent_color, plotly_template = "#334155", "#38BDF8", "plotly_dark"
else:
    bg_color, card_bg, text_primary, text_secondary = "#F8FAFC", "#FFFFFF", "#0F172A", "#475569"
    border_color, accent_color, plotly_template = "#E2E8F0", "#2563EB", "plotly_white"

st.markdown(f"""
<style>
    .stApp {{ background-color: {bg_color}; color: {text_primary}; }}
    .main-header {{ font-size: 2.2rem; font-weight: 800; color: {accent_color}; margin-bottom: 0.2rem; }}
    .sub-header {{ font-size: 1.05rem; color: {text_secondary}; margin-bottom: 1.5rem; }}
    .stButton>button {{
        background-color: {accent_color}; color: #FFFFFF !important; font-size: 1.05rem; font-weight: 600;
        border-radius: 8px; padding: 0.65rem 2rem; width: 100%; border: none; transition: all 0.3s ease;
    }}
    .stButton>button:hover {{ opacity: 0.9; transform: translateY(-1px); }}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_churn_model():
    if os.path.exists("best_churn_model.joblib"):
        return joblib.load("best_churn_model.joblib")
    try:
        model_path = hf_hub_download(repo_id="krish21may/Bank-Customer-Churn-4", filename="best_churn_model.joblib")
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_churn_model()

st.markdown('<div class="main-header">🛍️ E-Commerce Customer Churn Intelligence & MLOps Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Enterprise Shopper Platform: SHAP XAI • Counterfactual Recourse • Survival Timelines • Causal Uplift • Executive PDF</div>', unsafe_allow_html=True)

tab_single, tab_survival, tab_causal, tab_batch, tab_analytics, tab_drift, tab_pdf = st.tabs([
    "👤 Single Risk & SHAP XAI", 
    "⏳ Survival & Timeline",
    "🎯 Causal Uplift Matrix",
    "📁 Batch CSV Processor", 
    "📊 Portfolio Analytics",
    "⚖️ Fair Lending & Drift",
    "📄 Executive PDF Briefing"
])

# ==============================================================================
# TAB 1: SINGLE CUSTOMER PREDICTION & SHAP & COUNTERFACTUAL
# ==============================================================================
with tab_single:
    st.markdown("### 👤 Single Shopper Analysis & SHAP Explainability")
    
    preset = st.radio("Load Scenario Profile:", ["Custom Input", "⚠️ High Churn Risk Profile", "✅ Low Churn Risk Profile"], horizontal=True)
    
    if preset == "⚠️ High Churn Risk Profile":
        def_tenure, def_dist, def_hours, def_dev, def_sat, def_comp, def_hike, def_days, def_cash, def_tier, def_pay, def_gen, def_cat, def_mar = (
            2, 35, 2, 4, 1, 1, 12, 30, 80.0, 3, "COD", "Female", "Mobile Phone", "Single"
        )
    elif preset == "✅ Low Churn Risk Profile":
        def_tenure, def_dist, def_hours, def_dev, def_sat, def_comp, def_hike, def_days, def_cash, def_tier, def_pay, def_gen, def_cat, def_mar = (
            24, 8, 4, 2, 5, 0, 22, 3, 240.0, 1, "Credit Card", "Male", "Fashion", "Married"
        )
    else:
        def_tenure, def_dist, def_hours, def_dev, def_sat, def_comp, def_hike, def_days, def_cash, def_tier, def_pay, def_gen, def_cat, def_mar = (
            12, 15, 3, 3, 3, 0, 15, 8, 150.0, 1, "Debit Card", "Female", "Laptop & Accessory", "Married"
        )

    col_input, col_results = st.columns([1.1, 1], gap="large")

    with col_input:
        st.markdown("##### 📋 Shopper Behavior & Profile Inputs")
        c1, c2 = st.columns(2)
        with c1:
            Tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=def_tenure, key="s_tenure")
            WarehouseToHome = st.number_input("Warehouse to Home Dist (km)", min_value=1, max_value=150, value=def_dist, key="s_dist")
            HourSpendOnApp = st.slider("Hours Spent on App / Week", min_value=1, max_value=8, value=def_hours, key="s_hours")
            NumberOfDeviceRegistered = st.slider("Devices Registered", min_value=1, max_value=8, value=def_dev, key="s_dev")
            SatisfactionScore = st.slider("Satisfaction Score (1-5)", min_value=1, max_value=5, value=def_sat, key="s_sat")
            Complain = st.selectbox("Active Complaint?", ["No", "Yes"], index=1 if def_comp==1 else 0, key="s_comp")
            CityTier = st.selectbox("City Tier", [1, 2, 3], index=def_tier-1, key="s_tier")
        with c2:
            OrderAmountHikeFromlastYear = st.number_input("Order Amount Hike (%)", min_value=0, max_value=50, value=def_hike, key="s_hike")
            DaySinceLastOrder = st.number_input("Days Since Last Order", min_value=0, max_value=60, value=def_days, key="s_days")
            CashBackAmount = st.number_input("CashBack Amount ($)", min_value=0.0, max_value=500.0, value=float(def_cash), step=10.0, key="s_cash")
            PreferredPaymentMode = st.selectbox("Preferred Payment", ["Debit Card", "Credit Card", "E Wallet", "UPI", "COD"], index=0, key="s_pay")
            Gender = st.selectbox("Gender", ["Female", "Male"], index=0 if def_gen=="Female" else 1, key="s_gen")
            PreferedOrderCat = st.selectbox("Preferred Category", ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Others"], index=0, key="s_cat")
            MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], index=0, key="s_mar")

        predict_btn = st.button("🔍 Run Churn Risk & SHAP Analysis")

    single_input = pd.DataFrame([{
        'Tenure': Tenure, 'WarehouseToHome': WarehouseToHome, 'HourSpendOnApp': HourSpendOnApp,
        'NumberOfDeviceRegistered': NumberOfDeviceRegistered, 'SatisfactionScore': SatisfactionScore,
        'Complain': 1 if Complain == "Yes" else 0, 'OrderAmountHikeFromlastYear': OrderAmountHikeFromlastYear,
        'DaySinceLastOrder': DaySinceLastOrder, 'CashBackAmount': CashBackAmount, 'CityTier': CityTier,
        'PreferredPaymentMode': PreferredPaymentMode, 'Gender': Gender,
        'PreferedOrderCat': PreferedOrderCat, 'MaritalStatus': MaritalStatus
    }])

    with col_results:
        st.markdown("##### 📊 Prediction & SHAP Risk Attribution")
        
        if predict_btn or preset != "Custom Input":
            if model is not None:
                prob = float(model.predict_proba(single_input)[0, 1])
                threshold = 0.45
                is_churn = prob >= threshold
                
                st.session_state['single_prob'] = prob
                st.session_state['single_input'] = single_input
                
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=prob * 100,
                    number={'suffix': '%', 'font': {'size': 34, 'color': accent_color}},
                    title={'text': "Shopper Churn Risk Score", 'font': {'size': 18, 'color': text_primary}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#EF4444" if is_churn else "#10B981"},
                        'bgcolor': card_bg,
                        'steps': [
                            {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.2)'},
                            {'range': [35, 45], 'color': 'rgba(245, 158, 11, 0.2)'},
                            {'range': [45, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
                        ],
                        'threshold': {'line': {'color': "red", 'width': 3}, 'value': threshold * 100}
                    }
                ))
                fig_gauge.update_layout(template=plotly_template, height=210, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                shap_df = calculate_shap_contributions(model, single_input)
                fig_shap = px.bar(
                    shap_df, x='SHAP_Impact', y='Feature', orientation='h', color='Impact_Type',
                    color_discrete_map={'Increases Churn Risk ⚠️': '#EF4444', 'Decreases Churn Risk ✅': '#10B981'},
                    title="SHAP Feature Drivers", template=plotly_template
                )
                fig_shap.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=250)
                st.plotly_chart(fig_shap, use_container_width=True)
                
                st.markdown("##### 💡 Counterfactual 'What-If' Recourse")
                cf_scenarios = generate_counterfactual_scenarios(model, single_input)
                for sc in cf_scenarios:
                    with st.expander(f"{sc['Scenario_Name']} (Risk: {sc['Original_Risk_%']}% ➔ {sc['New_Risk_%']}%)"):
                        for act in sc['Actions_Required']:
                            st.write(f"- {act}")

                st.markdown("##### ✉️ LLM Generated Retention Copy")
                outreach = generate_llm_retention_outreach(single_input.iloc[0].to_dict(), prob, [])
                with st.expander("📄 View Customer Email & SMS Copy"):
                    st.code(outreach['Email_Body'], language="markdown")
                    st.caption(f"SMS Copy: {outreach['SMS_Copy']}")

# ==============================================================================
# TAB 2: SURVIVAL ANALYSIS & TIME-TO-CHURN
# ==============================================================================
with tab_survival:
    st.markdown("### ⏳ Survival Analysis & Retention Timeline")
    prob_val = st.session_state.get('single_prob', 0.48)
    surv_info = predict_survival_timeline(prob_val)
    
    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.metric("Expected Months Until Churn", f"{surv_info['Expected_Months_Until_Churn']} Months")
    c_m2.metric("6-Month Retention Probability", f"{surv_info['Prob_Survival_6M_%']}%")
    c_m3.metric("Hazard Risk Category", surv_info['Hazard_Risk_Category'])
    
    fig_surv = px.line(
        surv_info['Survival_Curve_DF'], x='Month', y='Survival_Probability_%',
        title="24-Month Shopper Survival Curve (Cox Hazard Projection)",
        markers=True, template=plotly_template
    )
    fig_surv.update_traces(line_color=accent_color, line_width=3)
    st.plotly_chart(fig_surv, use_container_width=True)

# ==============================================================================
# TAB 3: CAUSAL UPLIFT MATRIX
# ==============================================================================
with tab_causal:
    st.markdown("### 🎯 Causal Uplift Matrix & Campaign Optimization")
    test_df = pd.read_csv("Xtest.csv")
    probs = model.predict_proba(test_df)[:, 1] if model is not None else np.random.rand(len(test_df))
    test_df['Churn_Probability_%'] = probs * 100.0
    
    uplift_df = segment_causal_uplift(test_df)
    
    fig_up = px.histogram(
        uplift_df, x='Causal_Segment', color='Causal_Segment',
        title="Causal Uplift Customer Distribution", template=plotly_template
    )
    st.plotly_chart(fig_up, use_container_width=True)

# ==============================================================================
# TAB 4: BATCH CSV PROCESSOR
# ==============================================================================
with tab_batch:
    st.markdown("### 📁 Batch CSV Processor")
    uploaded_file = st.file_uploader("Upload Customer Dataset CSV", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        if model is not None:
            batch_probs = model.predict_proba(batch_df)[:, 1]
            batch_df['Churn_Risk_%'] = np.round(batch_probs * 100, 2)
            batch_df['Risk_Status'] = np.where(batch_probs >= 0.45, "HIGH RISK ⚠️", "SAFE ✅")
            st.dataframe(batch_df)
            
            csv_buf = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Scored CSV", csv_buf, "churn_predictions.csv", "text/csv")

# ==============================================================================
# TAB 5: PORTFOLIO ANALYTICS
# ==============================================================================
with tab_analytics:
    st.markdown("### 📊 Portfolio Risk & Financial Exposure Analytics")
    test_df = pd.read_csv("Xtest.csv")
    probs = model.predict_proba(test_df)[:, 1] if model is not None else np.random.rand(len(test_df))
    test_df['Churn_Risk_%'] = probs * 100.0
    
    c_a1, c_a2 = st.columns(2)
    with c_a1:
        fig_cat = px.box(test_df, x='PreferedOrderCat', y='Churn_Risk_%', color='PreferedOrderCat', title="Churn Risk by Order Category", template=plotly_template)
        st.plotly_chart(fig_cat, use_container_width=True)
    with c_a2:
        fig_pay = px.histogram(test_df, x='PreferredPaymentMode', color='PreferredPaymentMode', title="Payment Mode Distribution", template=plotly_template)
        st.plotly_chart(fig_pay, use_container_width=True)

    mc_res = run_monte_carlo_simulation(test_df)
    st.markdown("##### 🎲 Monte Carlo Revenue Attrition VaR Simulation (1,000 Trials)")
    st.info(f"Mean Revenue Loss: **${mc_res['Mean_Revenue_Loss_USD']:,.2f}** | 95% Confidence VaR Loss: **${mc_res['VaR_95_USD']:,.2f}**")

# ==============================================================================
# TAB 6: FAIR LENDING & DATA DRIFT
# ==============================================================================
with tab_drift:
    st.markdown("### ⚖️ Algorithmic Equity & Evidently AI Data Drift")
    test_df = pd.read_csv("Xtest.csv")
    probs = model.predict_proba(test_df)[:, 1] if model is not None else np.random.rand(len(test_df))
    test_df['Churn_Probability'] = probs
    
    fairness = run_fairness_audit(test_df, protected_attribute='Gender')
    st.markdown(f"##### Regulatory Compliance Status: **{fairness['Regulatory_Status']}**")
    st.json(fairness)
    
    drift_res = run_drift_analysis(test_df)
    st.markdown("##### Kolmogorov-Smirnov Feature Drift Analysis")
    st.json(drift_res)

# ==============================================================================
# TAB 7: EXECUTIVE PDF BRIEFING
# ==============================================================================
with tab_pdf:
    st.markdown("### 📄 Executive C-Suite PDF Governance Briefing")
    st.write("Generate and download a publication-ready PDF governance report.")
    
    if st.button("📄 Generate Executive PDF Briefing"):
        pdf_bytes = generate_executive_pdf_report()
        st.download_button(
            "📥 Download Executive Report PDF",
            pdf_bytes,
            file_name="Ecommerce_Churn_Executive_Briefing.pdf",
            mime="application/pdf"
        )
