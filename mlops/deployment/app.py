import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import joblib
import io
import sys
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

# Ensure root project directory is in python path for IDE extensions and runtime
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))

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

# Set Page Config
st.set_page_config(
    page_title="Bank Customer Churn Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Controls & Theme Selector
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/bank.png", width=70)
    st.title("Control Panel")
    
    st.markdown("### 🎨 Visual Theme")
    theme_mode = st.radio("Select Theme:", ["🌙 Dark Mode", "☀️ Light Mode"], index=0)
    is_dark = "Dark" in theme_mode

    st.markdown("---")
    st.markdown("### ⚙️ System Architecture")
    st.info("""
    - **Model**: Tuned XGBoost Classifier
    - **XAI**: SHAP Local Force Attribution
    - **Recourse**: DiCE Counterfactual AI
    - **Survival**: Cox Hazard 24M Curves
    - **Causal**: Uplift Segmentation
    - **Fairness**: 4/5th Rule ECOA Audit
    """)

# Dynamic Theme Tokens
if is_dark:
    bg_color, card_bg, text_primary, text_secondary = "#0F172A", "#1E293B", "#F8FAFC", "#94A3B8"
    border_color, accent_color, plotly_template = "#334155", "#38BDF8", "plotly_dark"
    card_shadow = "0 4px 12px rgba(0, 0, 0, 0.4)"
else:
    bg_color, card_bg, text_primary, text_secondary = "#F8FAFC", "#FFFFFF", "#0F172A", "#475569"
    border_color, accent_color, plotly_template = "#E2E8F0", "#2563EB", "plotly_white"
    card_shadow = "0 4px 6px -1px rgba(0, 0, 0, 0.05)"

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

# Load Model
@st.cache_resource
def load_churn_model():
    try:
        model_path = hf_hub_download(repo_id="krish21may/Bank-Customer-Churn-4", filename="best_churn_model.joblib")
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model from Hugging Face Hub: {e}")
        return None

model = load_churn_model()

# Header Banner
st.markdown('<div class="main-header">🏦 Bank Customer Churn Intelligence & MLOps Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Enterprise Decision Platform: SHAP XAI • Counterfactual Recourse • Survival Curves • Causal Uplift • Executive PDF</div>', unsafe_allow_html=True)

# Navigation Tabs
tab_single, tab_survival, tab_causal, tab_batch, tab_analytics, tab_drift, tab_pdf = st.tabs([
    "👤 Single Risk & SHAP XAI", 
    "⏳ Survival & Timeline",
    "🎯 Causal Uplift Matrix",
    "📁 Batch CSV Processor", 
    "📊 Portfolio Analytics",
    "⚖️ Fair Lending & Drift",
    "📄 Executive PDF Briefing"
])

REQUIRED_COLUMNS = ['CreditScore', 'Geography', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

# ==============================================================================
# TAB 1: SINGLE CUSTOMER PREDICTION & SHAP & COUNTERFACTUAL & LLM OUTREACH
# ==============================================================================
with tab_single:
    st.markdown("### 👤 Single Customer Analysis & SHAP Explainability")
    
    preset = st.radio("Load Scenario Profile:", ["Custom Input", "⚠️ High Churn Risk Profile", "✅ Low Churn Risk Profile"], horizontal=True)
    
    if preset == "⚠️ High Churn Risk Profile":
        def_credit, def_geo, def_age, def_tenure, def_balance, def_num_prod, def_card, def_active, def_salary = 590, "Germany", 52, 2, 125000.0, 1, "Yes", "No", 75000.0
    elif preset == "✅ Low Churn Risk Profile":
        def_credit, def_geo, def_age, def_tenure, def_balance, def_num_prod, def_card, def_active, def_salary = 750, "France", 28, 7, 45000.0, 2, "Yes", "Yes", 95000.0
    else:
        def_credit, def_geo, def_age, def_tenure, def_balance, def_num_prod, def_card, def_active, def_salary = 650, "France", 38, 5, 50000.0, 1, "Yes", "Yes", 60000.0

    col_input, col_results = st.columns([1.1, 1], gap="large")

    with col_input:
        st.markdown("##### 📋 Demographic & Financial Inputs")
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
        'CreditScore': CreditScore, 'Geography': Geography, 'Age': Age, 'Tenure': Tenure,
        'Balance': Balance, 'NumOfProducts': NumOfProducts,
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
                
                # Store in session state for tabs 2 & 3
                st.session_state['single_prob'] = prob
                st.session_state['single_input'] = single_input
                
                # Gauge Chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=prob * 100,
                    number={'suffix': '%', 'font': {'size': 34, 'color': accent_color}},
                    title={'text': "Churn Risk Score", 'font': {'size': 18, 'color': text_primary}},
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
                fig_gauge.update_layout(template=plotly_template, height=210, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # SHAP Bar Chart
                shap_df = calculate_shap_contributions(model, single_input)
                fig_shap = px.bar(
                    shap_df, x='SHAP_Impact', y='Feature', orientation='h', color='Impact_Type',
                    color_discrete_map={'Increases Churn Risk ⚠️': '#EF4444', 'Decreases Churn Risk ✅': '#10B981'},
                    title="SHAP Feature Force Drivers", template=plotly_template
                )
                fig_shap.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=250)
                st.plotly_chart(fig_shap, use_container_width=True)
                
                # Counterfactual What-If Recourse Scenarios
                st.markdown("##### 💡 Counterfactual 'What-If' Actionable Recourse")
                cf_scenarios = generate_counterfactual_scenarios(model, single_input)
                for sc in cf_scenarios:
                    with st.expander(f"{sc['Scenario_Name']} (Risk Drops: {sc['Original_Risk_%']}% ➔ {sc['New_Risk_%']}%)"):
                        for act in sc['Actions_Required']:
                            st.write(f"- {act}")

                # LLM Outreach
                st.markdown("##### ✉️ LLM Generated Personalized Retention Copy")
                outreach = generate_llm_retention_outreach(single_input.iloc[0].to_dict(), prob, [])
                with st.expander("📄 View Generated Customer Retention Email & SMS"):
                    st.code(outreach['Email_Body'], language="markdown")
                    st.caption(f"SMS Copy: {outreach['SMS_Copy']}")
            else:
                st.warning("Model not loaded.")
        else:
            st.info("👈 Set customer details on the left and click **'Run Churn Risk & SHAP Analysis'**.")

# ==============================================================================
# TAB 2: SURVIVAL ANALYSIS & TIME-TO-CHURN
# ==============================================================================
with tab_survival:
    st.markdown("### ⏳ Survival Analysis & Time-to-Churn Timeline")
    st.markdown("Models 24-month customer retention curves and predicts expected customer lifespan before attrition.")
    
    try:
        current_prob = st.session_state.get('single_prob', 0.65)
        surv_info = predict_survival_timeline(current_prob)
        
        s1, s2, s3 = st.columns(3)
        s1.metric("Est. Customer Lifespan", f"{surv_info['Expected_Months_Until_Churn']} Months")
        s2.metric("6-Month Survival Rate", f"{surv_info['Prob_Survival_6M_%']}%")
        s3.metric("Hazard Risk Category", surv_info['Hazard_Risk_Category'])
        
        st.markdown("##### 📈 24-Month Customer Survival Probability Curve")
        fig_surv = px.line(
            surv_info['Survival_Curve_DF'], x='Month', y='Survival_Probability_%',
            title="24-Month Retention Probability Trajectory", markers=True, template=plotly_template
        )
        fig_surv.update_traces(line_color=accent_color, line_width=3)
        fig_surv.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350)
        st.plotly_chart(fig_surv, use_container_width=True)
    except Exception as surv_err:
        st.warning(f"Survival analysis notice: {surv_err}")

# ==============================================================================
# TAB 3: CAUSAL ML & UPLIFT MATRIX
# ==============================================================================
with tab_causal:
    st.markdown("### 🎯 Causal ML & Uplift Campaign Matrix")
    st.markdown("Identifies **Persuadables** (customers who stay *only* if offered a retention incentive) to maximize marketing ROI.")
    
    try:
        sample_uplift_df = pd.DataFrame([
            {'CreditScore': 619, 'Geography': 'France', 'Age': 42, 'Balance': 85000.0, 'NumOfProducts': 1, 'IsActiveMember': 0, 'Churn_Probability': 0.62},
            {'CreditScore': 608, 'Geography': 'Spain', 'Age': 31, 'Balance': 45000.0, 'NumOfProducts': 2, 'IsActiveMember': 1, 'Churn_Probability': 0.18},
            {'CreditScore': 502, 'Geography': 'Germany', 'Age': 58, 'Balance': 150000.0, 'NumOfProducts': 3, 'IsActiveMember': 0, 'Churn_Probability': 0.88},
            {'CreditScore': 699, 'Geography': 'France', 'Age': 39, 'Balance': 0.0, 'NumOfProducts': 1, 'IsActiveMember': 0, 'Churn_Probability': 0.42}
        ])
        
        res_uplift = segment_causal_uplift(sample_uplift_df)
        
        u1, u2 = st.columns(2)
        with u1:
            st.markdown("##### 🎯 Causal Segment Breakdown")
            fig_causal = px.pie(res_uplift, names='Causal_Segment', title="Portfolio Uplift Segments", hole=0.4, template=plotly_template)
            fig_causal.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_causal, use_container_width=True)
            
        with u2:
            st.markdown("##### 💡 Campaign Targeting Guidelines")
            st.info("""
            - 🎯 **Persuadables**: High Uplift — Allocate 80% of retention campaign budget here.
            - 🔒 **Sure Things**: Low Churn Risk — Do not spend retention budget.
            - ❌ **Lost Causes**: Extremely high risk / inactive — Low campaign response.
            - ⚠️ **Sleeping Dogs**: Low risk — Do not disturb with unneeded emails.
            """)
    except Exception as uplift_err:
        st.warning(f"Causal uplift segmentation notice: {uplift_err}")

# ==============================================================================
# TAB 4: BATCH CSV PROCESSOR
# ==============================================================================
with tab_batch:
    st.markdown("### 📁 Batch Prediction Engine & CLV Optimizer")
    
    c_up, c_template = st.columns([2, 1], gap="large")
    with c_template:
        sample_df = pd.DataFrame([
            {'CreditScore': 619, 'Geography': 'France', 'Age': 42, 'Tenure': 2, 'Balance': 0.0, 'NumOfProducts': 1, 'HasCrCard': 1, 'IsActiveMember': 1, 'EstimatedSalary': 101348.88},
            {'CreditScore': 608, 'Geography': 'Spain', 'Age': 41, 'Tenure': 1, 'Balance': 83807.86, 'NumOfProducts': 1, 'HasCrCard': 0, 'IsActiveMember': 1, 'EstimatedSalary': 112542.58}
        ])
        buf = io.BytesIO()
        sample_df.to_csv(buf, index=False)
        st.download_button("📄 Download Sample CSV Template", buf.getvalue(), "sample_bank_customers.csv", "text/csv")
        
    with c_up:
        uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"], key="b_uploader")

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"File **'{uploaded_file.name}'** loaded ({len(batch_df)} records).")
            
            COLUMN_MAP = {
                'creditscore': 'CreditScore', 'credit_score': 'CreditScore', 'score': 'CreditScore',
                'geography': 'Geography', 'country': 'Geography', 'location': 'Geography',
                'age': 'Age',
                'tenure': 'Tenure', 'years': 'Tenure',
                'balance': 'Balance', 'account_balance': 'Balance',
                'numofproducts': 'NumOfProducts', 'num_of_products': 'NumOfProducts', 'products': 'NumOfProducts',
                'hascrcard': 'HasCrCard', 'has_cr_card': 'HasCrCard', 'credit_card': 'HasCrCard',
                'isactivemember': 'IsActiveMember', 'is_active_member': 'IsActiveMember', 'active': 'IsActiveMember',
                'estimatedsalary': 'EstimatedSalary', 'estimated_salary': 'EstimatedSalary', 'salary': 'EstimatedSalary'
            }

            DEFAULT_VALUES = {
                'CreditScore': 650, 'Geography': 'France', 'Age': 38, 'Tenure': 5,
                'Balance': 50000.0, 'NumOfProducts': 1, 'HasCrCard': 1, 'IsActiveMember': 1,
                'EstimatedSalary': 75000.0
            }

            # Map column names
            renamed_cols = {}
            for col in batch_df.columns:
                clean_col = str(col).strip().lower().replace(" ", "_")
                if clean_col in COLUMN_MAP:
                    renamed_cols[col] = COLUMN_MAP[clean_col]

            mapped_df = batch_df.rename(columns=renamed_cols).copy()

            # Identify missing required bank columns
            missing_cols = [c for c in REQUIRED_COLUMNS if c not in mapped_df.columns]
            if missing_cols:
                st.info(f"💡 Notice: File '{uploaded_file.name}' did not contain standard banking column headers. Missing fields ({missing_cols}) were auto-filled with baseline defaults to complete churn risk inference.")
                for m_col in missing_cols:
                    mapped_df[m_col] = DEFAULT_VALUES[m_col]

            proc_df = mapped_df[REQUIRED_COLUMNS].copy()

            # Ensure numeric data types & handle boolean/string values
            for b_col in ['HasCrCard', 'IsActiveMember']:
                proc_df[b_col] = proc_df[b_col].apply(lambda x: 1 if str(x).strip().lower() in ['1', 'yes', 'true'] else (0 if str(x).strip().lower() in ['0', 'no', 'false'] else 1))

            for num_col in ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']:
                proc_df[num_col] = pd.to_numeric(proc_df[num_col], errors='coerce').fillna(DEFAULT_VALUES[num_col])

            proc_df['Geography'] = proc_df['Geography'].astype(str).apply(lambda g: g if g in ['France', 'Germany', 'Spain'] else 'France')

            if model is not None:
                probs = model.predict_proba(proc_df)[:, 1]
                threshold = 0.45
                preds = (probs >= threshold).astype(int)
                
                res_df = batch_df.copy()
                # Ensure all required banking columns are present in res_df for downstream analytics
                for r_col in REQUIRED_COLUMNS:
                    if r_col not in res_df.columns and r_col in proc_df.columns:
                        res_df[r_col] = proc_df[r_col]

                res_df['Churn_Probability'] = probs
                res_df['Churn_Probability_%'] = np.round(probs * 100, 2)
                res_df['Risk_Status'] = np.where(preds == 1, 'HIGH RISK ⚠️', 'LOW RISK ✅')
                
                clvs = [calculate_clv(r['Balance'], r['EstimatedSalary'], r['NumOfProducts'], r['Tenure']) for _, r in proc_df.iterrows()]
                res_df['CLV_$'] = np.round(clvs, 2)
                
                st.session_state['batch_results'] = res_df
                st.session_state['batch_proc_df'] = proc_df
                
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
# TAB 5: PORTFOLIO VISUAL ANALYTICS
# ==============================================================================
with tab_analytics:
    st.markdown("### 📊 Portfolio Visual Analytics Dashboard")
    
    if 'batch_proc_df' in st.session_state and st.session_state['batch_proc_df'] is not None:
        df_ana = st.session_state['batch_proc_df'].copy()
        if 'batch_results' in st.session_state and st.session_state['batch_results'] is not None:
            for col_to_add in ['Churn_Probability_%', 'Churn_Probability', 'Risk_Status', 'CLV_$']:
                if col_to_add in st.session_state['batch_results'].columns:
                    df_ana[col_to_add] = st.session_state['batch_results'][col_to_add]
    else:
        df_ana = pd.DataFrame([
            {'CreditScore': 619, 'Geography': 'France', 'Age': 42, 'Tenure': 2, 'Balance': 0.0, 'NumOfProducts': 1, 'Churn_Probability_%': 62.4, 'Risk_Status': 'HIGH RISK ⚠️'},
            {'CreditScore': 608, 'Geography': 'Spain', 'Age': 41, 'Tenure': 1, 'Balance': 83807.86, 'NumOfProducts': 1, 'Churn_Probability_%': 18.2, 'Risk_Status': 'LOW RISK ✅'},
            {'CreditScore': 502, 'Geography': 'Germany', 'Age': 58, 'Tenure': 8, 'Balance': 159660.8, 'NumOfProducts': 3, 'Churn_Probability_%': 74.8, 'Risk_Status': 'HIGH RISK ⚠️'}
        ])

    ca1, ca2 = st.columns(2)
    with ca1:
        st.markdown("##### 📈 Demographics & Account Balance Cluster")
        try:
            fig_scatter = px.scatter(
                df_ana, x='Age', y='Balance', 
                size='Churn_Probability_%' if 'Churn_Probability_%' in df_ana.columns else None, 
                color='Risk_Status' if 'Risk_Status' in df_ana.columns else 'Geography', 
                template=plotly_template
            )
            fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_scatter, use_container_width=True)
        except Exception as scatter_err:
            st.warning(f"Notice: Visual analytics chart pending baseline data ({scatter_err}).")

    with ca2:
        st.markdown("##### 🎲 Monte Carlo Portfolio Attrition Simulation")
        try:
            mc_res = run_monte_carlo_simulation(df_ana)
            st.metric("95% Value-at-Risk (VaR) Deposit Loss", f"${mc_res['VaR_95_USD']:,.2f}")
            fig_hist = px.histogram(mc_res['Loss_Distribution'], nbins=30, title="Monte Carlo 1,000-Trial Attrition Distribution", template=plotly_template)
            fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_hist, use_container_width=True)
        except Exception as mc_err:
            st.warning(f"Notice: Monte Carlo simulation pending baseline metrics ({mc_err}).")

# ==============================================================================
# TAB 6: FAIR LENDING & EVIDENTLY DRIFT AUDIT
# ==============================================================================
with tab_drift:
    st.markdown("### ⚖️ Fair Lending Audit & Evidently Data Drift")
    
    try:
        if 'batch_results' in st.session_state and st.session_state['batch_results'] is not None:
            fairness_df = st.session_state['batch_results'].copy()
        elif 'batch_proc_df' in st.session_state and st.session_state['batch_proc_df'] is not None:
            fairness_df = st.session_state['batch_proc_df'].copy()
            if model is not None:
                probs = model.predict_proba(fairness_df)[:, 1]
                fairness_df['Churn_Probability'] = probs
                fairness_df['Churn_Probability_%'] = np.round(probs * 100, 2)
                fairness_df['Risk_Status'] = np.where(probs >= 0.45, 'HIGH RISK ⚠️', 'LOW RISK ✅')
        else:
            fairness_df = pd.DataFrame([
                {'CreditScore': 550, 'Geography': 'Germany', 'Age': 60, 'Tenure': 1, 'Balance': 180000.0, 'NumOfProducts': 3, 'HasCrCard': 1, 'IsActiveMember': 0, 'EstimatedSalary': 120000.0, 'Churn_Probability_%': 75.0, 'Risk_Status': 'HIGH RISK ⚠️'},
                {'CreditScore': 510, 'Geography': 'Spain', 'Age': 58, 'Tenure': 2, 'Balance': 195000.0, 'NumOfProducts': 4, 'HasCrCard': 0, 'IsActiveMember': 0, 'EstimatedSalary': 130000.0, 'Churn_Probability_%': 82.0, 'Risk_Status': 'HIGH RISK ⚠️'},
                {'CreditScore': 650, 'Geography': 'France', 'Age': 32, 'Tenure': 5, 'Balance': 45000.0, 'NumOfProducts': 2, 'HasCrCard': 1, 'IsActiveMember': 1, 'EstimatedSalary': 80000.0, 'Churn_Probability_%': 15.0, 'Risk_Status': 'LOW RISK ✅'}
            ])
            
        f_res = run_fairness_audit(fairness_df)
        d_res = run_drift_analysis(fairness_df)
        
        st.markdown("##### ⚖️ ECOA Fair Lending Disparate Impact Audit")
        st.metric("Disparate Impact Ratio (4/5th Rule)", f"{f_res['Disparate_Impact_Ratio']}", f_res['Regulatory_Status'])
        
        st.markdown("##### 📉 Evidently AI KS-Test Data Drift Status")
        st.metric("Overall Drift Status", "DRIFT DETECTED ⚠️" if d_res.get('Drift_Detected', False) else "HEALTHY (NO DRIFT) ✅", f"Drifted Share: {d_res.get('Drift_Share_%', 0.0)}%")
    except Exception as drift_err:
        st.warning(f"Fair lending & data drift analysis notice: {drift_err}")

# ==============================================================================
# TAB 7: EXECUTIVE PDF BRIEFING & FASTAPI DOCS
# ==============================================================================
with tab_pdf:
    st.markdown("### 📄 Download C-Suite Executive PDF Briefing")
    
    if 'batch_results' in st.session_state and st.session_state['batch_results'] is not None:
        pdf_df = st.session_state['batch_results']
    else:
        pdf_df = pd.DataFrame([
            {'CreditScore': 619, 'Geography': 'France', 'Age': 42, 'Balance': 85000.0, 'Churn_Probability_%': 62.4, 'Risk_Status': 'HIGH RISK ⚠️'},
            {'CreditScore': 608, 'Geography': 'Spain', 'Age': 41, 'Balance': 45000.0, 'Churn_Probability_%': 18.2, 'Risk_Status': 'LOW RISK ✅'}
        ])
        
    pdf_bytes = generate_executive_pdf_report(pdf_df)
    
    st.download_button(
        "📄 Download Publication-Ready Executive PDF Briefing",
        pdf_bytes,
        "executive_churn_intelligence_briefing.pdf",
        "application/pdf"
    )
    
    st.markdown("---")
    st.markdown("##### ⚡ FastAPI REST Microservice Documentation")
    st.code("uvicorn mlops.api.main:app --host 0.0.0.0 --port 8000 --reload", language="bash")
    st.info("Interactive OpenAPI / Swagger docs available at `http://localhost:8000/docs` when running.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #9CA3AF;'>Bank Customer Churn Intelligence Platform | XGBoost • SHAP • DiCE • Cox Hazard • Evidently • ReportLab</div>", unsafe_allow_html=True)
