import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# Set up page configuration
st.set_page_config(page_title="Student Academic Risk Predictor", layout="centered")

# WALLPAPER & LAYOUT ALIGNMENT CSS
st.markdown("""
    <style>
    /* Full-screen Wallpaper Background */
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?q=80&w=1964&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    /* Global Clean Typography Stack */
    html, body, [class*="css"], .stSlider, .stSelectbox, p, label {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Content Containers Layout */
    .config-block {
        background-color: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    div[data-testid="stForm"] {
        background-color: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 12px;
        padding: 28px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }
    
    /* Header typography elements */
    h1 {
        font-weight: 700 !important;
        color: #0F172A !important;
        font-size: 2.4rem !important;
    }
    h3 {
        font-weight: 600 !important;
        color: #1E293B !important;
        font-size: 1.25rem !important;
        margin-bottom: 12px !important;
    }
    
    /* FIX: Force Number Input layout controls to be small, clean, and contained */
    div[data-testid="stNumberInput"] > div {
        max-width: 140px !important;
    }
    div[data-testid="stNumberInput"] button {
        padding: 2px 10px !important;
        height: 32px !important;
        background-color: #F1F5F9 !important;
        color: #475569 !important;
        border: 1px solid #CBD5E1 !important;
    }
    
    /* Form Execution Button style profile */
    div[data-testid="stForm"] button[type="submit"] {
        background-color: #2563EB !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.75rem 2rem !important;
        border-radius: 6px !important;
        border: none !important;
        width: 100% !important;
        margin-top: 10px;
    }

    /* FIX: High-Visibility Result Popup Cards (White background text overlay) */
    .custom-popup-safe {
        background-color: #FFFFFF !important;
        border-left: 6px solid #16A34A !important;
        padding: 20px !important;
        border-radius: 8px !important;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.15);
        color: #1E293B !important;
        margin-top: 25px !important;
    }
    
    .custom-popup-risk {
        background-color: #FFFFFF !important;
        border-left: 6px solid #DC2626 !important;
        padding: 20px !important;
        border-radius: 8px !important;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.15);
        color: #1E293B !important;
        margin-top: 25px !important;
    }
    
    /* Hide specific heading elements link icons */
    .element-container:has(h1, h2, h3, h4) a {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Student Academic Risk Predictor")
st.markdown("Use this assessment panel to review individual student progress trends and identify early risk vectors.")

# Cache model training so it only runs once at startup
@st.cache_resource
def train_model_live():
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
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
    model.fit(X_train_proc, y_train)
    return preprocessor, model, X.columns.tolist(), num_cols

preprocessor, model, features, num_cols = train_model_live()

# Configuration Block Container
#st.markdown('<div class="config-block">', unsafe_allow_html=True)
st.write("### Module Configuration")

selected_module = st.selectbox(
    "Select Module",
    options=[
        "COM 763: Advanced Machine Learning",
        "COM 742: Enterprise Data Systems",
        "COM 711: Software Engineering Foundations",
        "COM 705: Artificial Intelligence Principles"
    ]
)

syllabus_blueprints = {
    "COM 763: Advanced Machine Learning": [
        "Coursework 1: Exploratory Data Analysis Portfolio",
        "Coursework 2: Model Training & Hyperparameter GridSearch",
        "Coursework 3: Streamlit Deployment & Technical Report",
        "Final Examination Component"
    ],
    "COM 742: Enterprise Data Systems": [
        "Coursework 1: Relational Schema & SQL Design",
        "Coursework 2: NoSQL Database Scaling Lab",
        "Coursework 3: Distributed Data Infrastructure Project",
        "Final Examination Component"
    ],
    "COM 711: Software Engineering Foundations": [
        "Coursework 1: Requirements Specification & UML Diagrams",
        "Coursework 2: Object-Oriented Programming Core Build",
        "Coursework 3: System Testing & CI/CD Validation",
        "Final Examination Component"
    ],
    "COM 705: Artificial Intelligence Principles": [
        "Coursework 1: Search Algorithm Optimization Lab",
        "Coursework 2: Neural Network Implementation Project",
        "Coursework 3: Ethical AI Case Study Defense",
        "Final Examination Component"
    ]
}

short_name = selected_module.split(':')[0]
active_syllabus = syllabus_blueprints[selected_module]
max_components = len(active_syllabus)

col_config1, col_config2 = st.columns(2)
with col_config1:
    total_assignments = st.selectbox(
        f"Total assessments in syllabus ({short_name})",
        options=list(range(2, max_components + 1)),
        index=2
    )
with col_config2:
    completed_assignments = st.selectbox(
        "Assessments completed by student so far",
        options=list(range(1, total_assignments + 1)),
        index=1
    )
st.markdown('</div>', unsafe_allow_html=True)

# Main Form Block
with st.form("input_marks_form"):
    st.write("### Assessment Marks (0 - 20 range)")
    
    grades = []
    for i in range(completed_assignments):
        assignment_label = active_syllabus[i]
        score = st.slider(assignment_label, min_value=0, max_value=20, value=10)
        grades.append(score)
            
    st.markdown("---")
    st.write("### Student Background & Attendance")
    
    # FIX: Clean, explicitly separated column widths to secure true single-line alignment
    col_a, col_b, col_c = st.columns([1, 1.2, 1.3])
    with col_a:
        absences = st.number_input("Total Absences", min_value=0, max_value=100, value=2, step=1)
    with col_b:
        failures = st.selectbox("Past Module Failures", options=[0, 1, 2, 3, 4])
    with col_c:
        study_time_opts = {1: "< 2 Hours", 2: "2 - 5 Hours", 3: "5 - 10 Hours", 4: "> 10 Hours"}
        studytime = st.selectbox("Weekly Independent Study Time", 
                                 options=list(study_time_opts.keys()), 
                                 format_func=lambda x: study_time_opts[x])
        
    submit_button = st.form_submit_button("Calculate Risk Prediction")

# POP-UP BOX DISPLAY MAPPING
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
    input_dict['G1'] = [g1_mapped]
    input_dict['G2'] = [g2_mapped]
    input_dict['absences'] = [absences]
    input_dict['failures'] = [failures]
    input_dict['studytime'] = [studytime]
    
    input_df = pd.DataFrame(input_dict)
    processed_input = preprocessor.transform(input_df)
    
    pred = model.predict(processed_input)[0]
    prob = model.predict_proba(processed_input)[0][1]
    
    st.toast("Analysis complete!", icon="🔄")
    
    # FIX: Injecting clean custom white high-contrast banner blocks for maximum text legibility
    if pred == 1:
        st.markdown(f"""
        <div class="custom-popup-risk">
            <h4 style="color: #DC2626; margin: 0 0 8px 0; font-weight:700;">⚠️ Prediction Verdict: Student is At-Risk</h4>
            <p style="margin: 0; font-size: 0.95rem; line-height: 1.5; color: #334155;">
                The evaluation engine has flagged this student profile with an estimated failure risk probability of <strong>{prob:.1%}</strong>. 
                Immediate outreach or targeted support sessions for <strong>{short_name}</strong> are highly recommended.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="custom-popup-safe">
            <h4 style="color: #16A34A; margin: 0 0 8px 0; font-weight:700;">✅ Prediction Verdict: Student is Safe</h4>
            <p style="margin: 0; font-size: 0.95rem; line-height: 1.5; color: #334155;">
                The evaluation engine predicts this student is currently on track to clear all pass requirements. 
                The calculated module failure risk is low (<strong>{prob:.1%}</strong>).
            </p>
        </div>
        """, unsafe_allow_html=True)
