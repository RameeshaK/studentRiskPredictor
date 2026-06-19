import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# Set up page configuration
st.set_page_config(page_title="Dynamic Academic Risk Engine", layout="centered")

st.title("🎓 University Student Academic Risk Predictor")
st.markdown("""
This enterprise decision-support tool dynamically adapts to specific module syllabi structures 
to identify students who require early academic intervention.
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

# Enterprise configuration form
with st.form("dynamic_assessment_form"):
    st.subheader("📋 Syllabus & Student Profile Mapping")
    
    # 1. Module Selector
    selected_module = st.selectbox(
        "Select University Module",
        options=[
            "COM 763: Advanced Machine Learning",
            "COM 742: Enterprise Data Systems",
            "COM 711: Software Engineering Foundations",
            "COM 705: Artificial Intelligence Principles"
        ]
    )
    
    # 2. Dynamic Assessment Configuration
    short_name = selected_module.split(':')[0]
    total_assignments = st.selectbox(
        f"Total Continuous Assessments planned for {short_name}",
        options=[2, 3, 4],
        index=1, # Default to 3 assignments
        help="Select how many total coursework components make up this specific module's syllabus."
    )
    
    completed_assignments = st.selectbox(
        "Number of Assignments currently completed by the student",
        options=list(range(1, total_assignments + 1)),
        index=0,
        help="How many grades are currently available to evaluate?"
    )
    
    st.markdown("---")
    st.write("📊 **Enter Available Coursework Marks (0 - 20 Marks per Assignment)**")
    
    # Dynamically generate sliders based on the number of completed assignments selected
    grades = []
    cols = st.columns(completed_assignments)
    for i in range(completed_assignments):
        with cols[i]:
            score = st.slider(f"Assignment {i+1} Mark", 0, 20, 10)
            grades.append(score)
            
    st.markdown("---")
    st.write("🏃‍♂️ **Behavioral & Study Factors**")
    
    # Behavioral Background Layout
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        absences = st.number_input("Total Class Absences", min_value=0, max_value=100, value=2, step=1)
    with col_b:
        failures = st.selectbox("Previous Module Failures", options=[0, 1, 2, 3, 4])
    with col_c:
        study_time_opts = {1: "< 2 Hours", 2: "2 - 5 Hours", 3: "5 - 10 Hours", 4: "> 10 Hours"}
        studytime = st.selectbox("Weekly Study Dedication", 
                                 options=list(study_time_opts.keys()), 
                                 format_func=lambda x: study_time_opts[x])
        
    submit_button = st.form_submit_button("Execute Risk Analysis", use_container_width=True)

# Post-submission mapping pipeline
if submit_button:
    # Safely map dynamic grades back to the model's static G1 and G2 expectations:
    # If 1 assignment done: use it for G1 and G2. 
    # If 2 or more assignments done: average the early ones for G1, use the latest for G2.
    if len(grades) == 1:
        g1_mapped = grades[0]
        g2_mapped = grades[0]
    else:
        g1_mapped = int(np.mean(grades[:-1]))
        g2_mapped = grades[-1]
        
    # Construct feature matrix matching original dataset schema
    input_dict = {col: [0] if col in num_cols else ['M'] for col in features}
    input_dict['G1'] = [g1_mapped]
    input_dict['pattern_g2'] = [g2_mapped] # Keeping the mapping aligned
    input_dict['G2'] = [g2_mapped]
    input_dict['absences'] = [absences]
    input_dict['failures'] = [failures]
    input_dict['studytime'] = [studytime]
    
    input_df = pd.DataFrame(input_dict)
    processed_input = preprocessor.transform(input_df)
    
    pred = model.predict(processed_input)[0]
    prob = model.predict_proba(processed_input)[0][1]
    
    st.subheader("🔍 Analysis Verdict")
    if pred == 1:
        st.error(f"⚠️ **Intervention Flagged:** This student is predicted as **AT-RISK** of failing {short_name}. (Risk Probability: {prob:.2%})")
    else:
        st.success(f"✅ **Clearance:** This student is predicted as **SAFE** to pass {short_name}. (Failure Risk: {prob:.2%})")
