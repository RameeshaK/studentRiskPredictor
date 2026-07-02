import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from streamlit_option_menu import option_menu

# Page Config
st.set_page_config(page_title="Student Academic Risk Predictor", layout="centered")

# Global UI Customization
st.markdown("""
    <style>
    .stApp {
        background-image: linear-gradient(rgba(255, 255, 255, 0.85), rgba(255, 255, 255, 0.85)), 
                          url("https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?q=80&w=1964&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    html, body, [class*="css"], .stSlider, .stSelectbox, p, label {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .module-config-card {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    div[data-testid="stForm"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px;
        padding: 28px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    
    label, .stWidgetLabel p {
        color: #0F172A !important;
        font-weight: 500 !important;
    }
    
    h1 {
        font-weight: 700 !important;
        color: #0F172A !important;
        font-size: 2.4rem !important;
        margin-bottom: 4px !important;
    }
    h3 {
        font-weight: 600 !important;
        color: #0F172A !important;
        font-size: 1.25rem !important;
        margin-bottom: 16px !important;
        margin-top: 0px !important;
    }
    
    div[data-testid="stNumberInput"] {
        max-width: 100% !important;
    }
    
    div[data-testid="stForm"] button[type="submit"] {
        background-color: #2563EB !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.75rem 2rem !important;
        border-radius: 6px !important;
        border: none !important;
        width: 100% !important;
        margin-top: 10px !important;
    }

    .custom-popup-safe {
        background-color: #FFFFFF !important;
        border-left: 6px solid #16A34A !important;
        padding: 20px !important;
        border-radius: 8px !important;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.15);
        color: #0F172A !important;
        margin-top: 25px !important;
    }
    
    .custom-popup-risk {
        background-color: #FFFFFF !important;
        border-left: 6px solid #DC2626 !important;
        padding: 20px !important;
        border-radius: 8px !important;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.15);
        color: #0F172A !important;
        margin-top: 25px !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    
    [data-testid="stSidebarCollapsedControl"], 
    button[data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }
    
    [data-testid="stSidebarNav"] {
        display: none !important;
    }

    .element-container:has(h1, h2, h3, h4) a {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# Experimental Multi-Model Engine Pipeline
@st.cache_resource
def train_models_live():
    url = "https://raw.githubusercontent.com/KunjalJethwani/StudentPerformance/master/student-por.csv"
    df = pd.read_csv(url, sep=';')
    df['at_risk'] = np.where(df['G3'] < 10, 1, 0)
    df = df.drop(columns=['G3'])
    X = df.drop(columns=['at_risk'])
    y = df['at_risk']
    
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
        ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # Model 1: Random Forest (Primary Selection)
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
    rf_model.fit(X_train_proc, y_train)
    
    # Model 2: Logistic Regression (Comparative Engine Baseline)
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr_model.fit(X_train_proc, y_train)
    
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_cols = cat_encoder.get_feature_names_out(cat_cols).tolist()
    all_features = num_cols + encoded_cat_cols
    
    models_dict = {"Random Forest Classifier": rf_model, "Logistic Regression": lr_model}
    metrics_log = {}
    
    for name, model_obj in models_dict.items():
        preds = model_obj.predict(X_test_proc)
        metrics_log[name] = {
            "Accuracy": f"{accuracy_score(y_test, preds):.1%}",
            "Recall": f"{recall_score(y_test, preds):.1%}",
            "Precision": f"{precision_score(y_test, preds):.1%}",
            "F1-Score": f"{f1_score(y_test, preds):.1%}",
            "y_pred": preds
        }
        
    return preprocessor, models_dict, metrics_log, X.columns.tolist(), num_cols, X_test_proc, y_test, all_features

preprocessor, models_dict, metrics_log, features, num_cols, X_test_proc, y_test, all_features = train_models_live()

# Sidebar Setup
with st.sidebar:
    st.markdown("<br><h2 style='color: #0F172A; font-weight: 700; margin-left: 10px; font-size: 1.6rem; font-family: \"Inter\", sans-serif;'>Menu Panel</h2><hr style='border-color: #E2E8F0;'>", unsafe_allow_html=True)
    page = option_menu(
        menu_title=None,
        options=["Home", "Evaluation Metrics"],
        icons=["house", "bar-chart-line"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#FFFFFF"},
            "icon": {"color": "#475569", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "4px 10px", "color": "#475569", "border-radius": "8px", "font-family": "'Inter', sans-serif"},
            "nav-link-selected": {"background-color": "#F1F5F9", "color": "#2563EB !important", "font-weight": "600"}
        }
    )

if page == "Home":
    st.title("Student Academic Risk Predictor")
    st.markdown("<p style='color: #1E293B; font-size: 1.1rem; margin-bottom: 24px; font-weight: 400;'>Use this assessment panel to review individual student progress trends and identify early risk vectors.</p>", unsafe_allow_html=True)

    st.write("### Module & Model Selection Blueprint")
    
    col_mod1, col_mod2 = st.columns(2)
    with col_mod1:
        selected_module = st.selectbox(
            "Select Module",
            options=["COM 763: Advanced Machine Learning", "COM 742: Enterprise Data Systems", "COM 711: Software Engineering Foundations", "COM 705: Artificial Intelligence Principles"]
        )
    with col_mod2:
        chosen_algorithm = st.selectbox(
            "Active Classification Model",
            options=["Random Forest Classifier", "Logistic Regression"]
        )
    
    syllabus_blueprints = {
        "COM 763: Advanced Machine Learning": ["Coursework 1: Exploratory Data Analysis Portfolio", "Coursework 2: Model Training & Hyperparameter GridSearch", "Coursework 3: Streamlit Deployment & Technical Report", "Final Examination Component"],
        "COM 742: Enterprise Data Systems": ["Coursework 1: Relational Schema & SQL Design", "Coursework 2: NoSQL Database Scaling Lab", "Coursework 3: Distributed Data Infrastructure Project", "Final Examination Component"],
        "COM 711: Software Engineering Foundations": ["Coursework 1: Requirements Specification & UML Diagrams", "Coursework 2: Object-Oriented Programming Core Build", "Coursework 3: System Testing & CI/CD Validation", "Final Examination Component"],
        "COM 705: Artificial Intelligence Principles": ["Coursework 1: Search Algorithm Optimization Lab", "Coursework 2: Neural Network Implementation Project", "Coursework 3: Ethical AI Case Study Defense", "Final Examination Component"]
    }

    short_name = selected_module.split(':')[0]
    active_syllabus = syllabus_blueprints[selected_module]
    max_components = len(active_syllabus)

    col_config1, col_config2 = st.columns(2)
    with col_config1:
        total_assignments = st.selectbox(f"Total assessments in syllabus ({short_name})", options=list(range(1, max_components + 1)), index=1)
    with col_config2:
        completed_assignments = st.selectbox("Assessments completed by student so far", options=list(range(1, total_assignments + 1)), index=0)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.form("input_marks_form"):
        st.write("### Assessment Marks (0 - 20 range)")
        grades = []
        for i in range(completed_assignments):
            assignment_label = active_syllabus[i]
            score = st.slider(assignment_label, min_value=0, max_value=20, value=10)
            grades.append(score)
                
        st.markdown("---")
        st.write("### Student Background & Attendance")
        
        col_a, col_b, col_c = st.columns([1, 1.2, 1.3])
        with col_a:
            absences = st.number_input("Total Absences", min_value=0, max_value=100, value=2, step=1)
        with col_b:
            failures = st.selectbox("Past Module Failures", options=[0, 1, 2, 3, 4])
        with col_c:
            study_time_opts = {1: "< 2 Hours", 2: "2 - 5 Hours", 3: "5 - 10 Hours", 4: "> 10 Hours"}
            studytime = st.selectbox("Weekly Independent Study Time", options=list(study_time_opts.keys()), format_func=lambda x: study_time_opts[x])
            
        submit_button = st.form_submit_button("Calculate Risk Prediction")

    if submit_button:
        if len(grades) == 1:
            g1_mapped = grades[0]
            g2_mapped = grades[0]
        elif len(grades) == 2:
            g1_mapped = grades[0]
            g2_mapped = grades[1]
        else:
            g1_mapped = int(np.mean(grades[:-1]))
            g2_mapped = grades[-1]
            
        input_dict = {col: [0] if col in num_cols else ['M'] for col in features}
        
        # Mapping base values
        input_dict['age'] = [17]
        input_dict['G1'] = [g1_mapped]
        input_dict['G2'] = [g2_mapped]
        input_dict['absences'] = [absences]
        input_dict['failures'] = [failures]
        input_dict['studytime'] = [studytime]
        
        input_df = pd.DataFrame(input_dict)
        processed_input = preprocessor.transform(input_df)
        
        active_model = models_dict[chosen_algorithm]
        pred = active_model.predict(processed_input)[0]
        prob = active_model.predict_proba(processed_input)[0][1]
        
        st.toast(f"Analysis complete!", icon="🔄")
        
        if pred == 1:
            st.markdown(f"""
            <div class="custom-popup-risk">
                <h4 style="color: #DC2626; margin: 0 0 8px 0; font-weight:700;">⚠️ Prediction Verdict: Student is At-Risk</h4>
                <p style="margin: 0; font-size: 0.95rem; line-height: 1.5; color: #0F172A;">
                    The evaluation engine has flagged this student profile with an estimated failure risk probability of <strong>{prob:.1%}</strong> via <strong>{chosen_algorithm}</strong>. 
                    Immediate outreach or targeted support sessions for <strong>{short_name}</strong> are highly recommended.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="custom-popup-safe">
                <h4 style="color: #16A34A; margin: 0 0 8px 0; font-weight:700;">✅ Prediction Verdict: Student is Safe</h4>
                <p style="margin: 0; font-size: 0.95rem; line-height: 1.5; color: #0F172A;">
                    The evaluation engine predicts this student is currently on track to clear all pass requirements using <strong>{chosen_algorithm}</strong>. 
                    The calculated module failure risk is low (<strong>{prob:.1%}</strong>).
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.write("<br>", unsafe_allow_html=True)
        
        # REFACTORED USER-FRIENDLY PIPELINE PORTAL
        with st.expander("🔍 View Preprocessed Real-Time Input Feature Vector"):
            st.markdown("### Algorithmic Feature Vector Pipeline")
            st.write("This table shows the feature values processed by the pipeline engine:")
            
            display_data = {
                "Feature Name": [
                    "First Assessment Grade (G1)", 
                    "Second Assessment Grade (G2)", 
                    "Total Semester Absences", 
                    "Historical Module Failures", 
                    "Independent Study Allocation"
                ],
                "Submitted Raw Input": [
                    f"{g1_mapped} / 20",
                    f"{g2_mapped} / 20",
                    f"{absences} days",
                    f"{failures} modules",
                    study_time_opts[studytime]
                ],
                "Model Feature Assignment": [
                    "Continuous Numerical Vector",
                    "Continuous Numerical Vector",
                    "Continuous Numerical Vector",
                    "Ordinal Class Identifier",
                    "Categorical Categorized Array"
                ]
            }
            
            summary_df = pd.DataFrame(display_data)
            st.table(summary_df)
            
            st.write("**Model Engine Raw Input Representation Layer:**")
            transformed_df = pd.DataFrame(processed_input, columns=all_features)
            st.dataframe(transformed_df.loc[:, (transformed_df != 0).any(axis=0)])

elif page == "Evaluation Metrics":
    st.title("Model Evaluation Engine Metrics")
    st.markdown("<p style='color: #1E293B; font-size: 1.1rem; margin-bottom: 24px; font-weight: 400;'>Review live data-driven trends, algorithmic breakdown matrices, and feature extraction weights.</p>", unsafe_allow_html=True)
    
    st.write("### Model Validation Metrics (Live Calculated)")
    
    rf_m = metrics_log["Random Forest Classifier"]
    lr_m = metrics_log["Logistic Regression"]
    
    comparison_table = {
        "Metric Parameter": ["Overall Accuracy", "Sensitivity (Recall)", "Precision", "F1-Score"],
        "Random Forest Engine": [rf_m["Accuracy"], rf_m["Recall"], rf_m["Precision"], rf_m["F1-Score"]],
        "Logistic Regression Engine": [lr_m["Accuracy"], lr_m["Recall"], lr_m["Precision"], lr_m["F1-Score"]]
    }
    st.table(pd.DataFrame(comparison_table))
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.write("### Visualized Algorithmic Diagnostics Matrix")
    
    chosen_chart_model = st.selectbox("Select Target Model for Visualizations", options=["Random Forest Classifier", "Logistic Regression"])
    
    plot_col1, plot_col2 = st.columns(2)
    with plot_col1:
        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 4))
        cm = confusion_matrix(y_test, metrics_log[chosen_chart_model]["y_pred"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Safe', 'At-Risk'])
        disp.plot(ax=ax_cm, cmap='Blues', values_format='d', colorbar=False)
        ax_cm.set_title(f'Confusion Matrix\n({chosen_chart_model})', fontsize=10, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_cm, use_container_width=True)
    
    with plot_col2:
        fig_fi, ax_fi = plt.subplots(figsize=(4.5, 4))
        if chosen_chart_model == "Random Forest Classifier":
            importances = models_dict["Random Forest Classifier"].feature_importances_
            title_label = 'Top Random Forest Feature Importances'
        else:
            importances = np.abs(models_dict["Logistic Regression"].coef_[0])
            title_label = 'Top Logistic Regression Weights'
            
        feat_importances = pd.Series(importances, index=all_features).sort_values(ascending=False).head(8)
        sns.barplot(x=feat_importances.values, y=feat_importances.index, ax=ax_fi, palette='Blues_r')
        ax_fi.set_title(title_label, fontsize=10, fontweight='bold')
        ax_fi.set_xlabel('Relative Score Metric', fontsize=9)
        ax_fi.tick_params(axis='both', labelsize=9)
        plt.tight_layout()
        st.pyplot(fig_fi, use_container_width=True)
        
    st.markdown('</div>', unsafe_allow_html=True)
