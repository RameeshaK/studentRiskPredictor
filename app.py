import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# Set up clean page configuration
st.set_page_config(page_title="Student Academic Risk Predictor", layout="centered")

# Custom CSS Injection to change the entire layout look and color profile
st.markdown("""
    <style>
    /* Change base typography across the entire application */
    html, body, [class*="css"], .stSlider, .stSelectbox, p, label {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }
    
    /* App Canvas Background */
    .stApp {
        background-color: #FFFFFF;
    }
    
    /* Title typography */
    h1 {
        font-weight: 700 !important;
        color: #0F172A !important;
        font-size: 2.25rem !important;
        letter-spacing: -0.025em !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Section Headers */
    h3 {
        font-weight: 600 !important;
        color: #1E293B !important;
        font-size: 1.3rem !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Label text formatting */
    label, .stWidgetLabel p {
        font-weight: 500 !important;
        color: #475569 !important;
        font-size: 0.95rem !important;
    }
    
    /* Custom container styling for Module Configuration block */
    .config-block {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Custom styling for the main interactive Form container */
    div[data-testid="stForm"] {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 28px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    
    /* Customizing the Primary Form Action Button */
    div[data-testid="stForm"] button {
        background-color: #2563EB !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 0.6rem 2rem !important;
        border-radius: 6px !important;
        transition: background-color 0.2s ease !important;
    }
    
    div[data-testid="stForm"] button:hover {
        background-color: #1D4ED8 !important;
    }

    /* Customizing Sliders to match the Indigo theme color */
    .stSlider [data-baseweb="slider"] > div > div {
        background: #2563EB !important;
    }
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background-color: #2563EB !important;
        border: 2px solid #FFFFFF !important;
    }
    
    /* Clear and clean custom metric outputs */
    .result-box-safe {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        color: #166534;
        padding: 16px;
        border-radius: 8px;
        font-weight: 500;
    }
    
    .result-box-risk {
        background-color: #FEF2F2;
        border: 1px solid #FEE2E2;
        color: #991B1B;
        padding: 16px;
        border-radius: 8px;
        font-weight: 500;
    }

    /* Hide the automatic anchor link icon next to headers */
    .element-container:has(h1, h2, h3, h4) a {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Student Academic Risk Predictor")
st.markdown("""
This decision-support panel helps module coordinators identify students who may need early academic support 
based on their continuous assessment performance and attendance trends.
""")

# Cache model training so it only runs once at startup
@st.cache_resource
def train_model_live():
    url = "https://raw.githubusercontent.com/KunjalJethwani/StudentPerformance/master/student-por.csv"
    df = pd.read_csv(url, sep=';')
    
    # Target engineering (1 = At Risk [G3 < 10], 0 = Safe)
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

# Using a distinct div wrapper to apply the configuration styling background block
st.markdown('<div class="config-block">', unsafe_allow_html=True)
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

# Form block for user data entries with custom colored button and layouts
with st.form("input_marks_form"):
    st.write("### Assessment Marks (0 - 20 range)")
    
    grades = []
    for i in range(completed_assignments):
        assignment_label = active_syllabus[i]
        score = st.slider(assignment_label, min_value=0, max_value=20, value=10)
        grades.append(score)
            
    st.markdown("---")
    st.write("### Student Background & Attendance")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        absences = st.number_input("Total Absences", min_value=0, max_value=100, value=2, step=1)
    with col_b:
        failures = st.selectbox("Past Module Failures", options=[0, 1, 2, 3, 4],
                                help="Number of modules failed in previous semesters.")
    with col_c:
        study_time_opts = {1: "< 2 Hours", 2: "2 - 5 Hours", 3: "5 - 10 Hours", 4: "> 10 Hours"}
        studytime = st.selectbox("Weekly Independent Study Time", 
                                 options=list(study_time_opts.keys()), 
                                 format_func=lambda x: study_time_opts[x])
        
    submit_button = st.form_submit_button("Calculate Risk Prediction", use_container_width=True)

# Post-submission processing
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
    
    st.write("### Prediction Result")
    if pred == 1:
        st.markdown(f"""
        <div class="result-box-risk">
            <strong>Status Flagged: At-Risk of failure</strong><br>
            Estimated failure risk probability is <strong>{prob:.1%}</strong>. 
            It is recommended to flag this student for academic review or practical support workshops in {short_name}.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box-safe">
            <strong>Status Clear: On track to pass</strong><br>
            The student is currently projected to successfully complete the module requirements. (Failure risk is low: <strong>{prob:.1%}</strong>)
        </div>
        """, unsafe_allow_html=True)
