import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# Configure a professional page layout
st.set_page_config(page_title="Student Risk Predictor", layout="centered")

st.title("🎓 Student Academic Performance Predictor")
st.markdown("""
This decision-support tool assists educators in identifying students who may require early academic intervention. 
Complete the profiles below to evaluate the risk status.
""")

# Cache model training so it only runs once at startup
@st.cache_resource
def train_model_live():
    url = "https://raw.githubusercontent.com/KunjalJethwani/StudentPerformance/master/student-por.csv"
    df = pd.read_csv(url, sep=';')
    
    # Target engineering
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

# Form layout to group inputs cleanly
with st.form("prediction_form"):
    st.subheader("📊 Student Profile Assessment")
    
    # Row 1: Academic Grades
    col1, col2 = st.columns(2)
    with col1:
        g1 = st.slider("First Term Grade (0 - 20 Marks)", 0, 20, 10, 
                       help="Student's mid-term examination score.")
    with col2:
        g2 = st.slider("Second Term Grade (0 - 20 Marks)", 0, 20, 10, 
                       help="Student's subsequent assessment score.")
        
    st.markdown("---")
    
    # Row 2: Behavioral Background
    col3, col4, col5 = st.columns(3)
    with col3:
        absences = st.number_input("Total Class Absences", min_value=0, max_value=100, value=4, step=1)
    with col4:
        failures = st.selectbox("Previous Course Failures", 
                                options=[0, 1, 2, 3, 4], 
                                format_func=lambda x: "None" if x == 0 else f"{x} Classes")
    with col5:
        # Convert numeric choices to friendly text descriptions
        study_time_opts = {1: "< 2 Hours", 2: "2 - 5 Hours", 3: "5 - 10 Hours", 4: "> 10 Hours"}
        studytime = st.selectbox("Weekly Study Dedication", 
                                 options=list(study_time_opts.keys()), 
                                 format_func=lambda x: study_time_opts[x])
        
    # Center the button submission
    submit_button = st.form_submit_button("Analyze Risk Status", use_container_width=True)

# Process results outside the form container
if submit_button:
    # Construct processing dataframe mapping original schema names
    input_dict = {col: [0] if col in num_cols else ['M'] for col in features}
    input_dict['G1'] = [g1]
    input_dict['G2'] = [g2]
    input_dict['absences'] = [absences]
    input_dict['failures'] = [failures]
    input_dict['studytime'] = [studytime]
    
    input_df = pd.DataFrame(input_dict)
    processed_input = preprocessor.transform(input_df)
    
    pred = model.predict(processed_input)[0]
    prob = model.predict_proba(processed_input)[0][1]
    
    st.subheader("🔍 Analysis Verdict")
    if pred == 1:
        st.error(f"⚠️ **Action Required:** This student is flagged as **AT-RISK** of failing. (Risk Confidence: {prob:.2%})")
        st.markdown("""
        **Recommended Interventions:**
        * Schedule a mandatory academic counseling check-in.
        * Enroll the student in peer-assisted tutoring sessions.
        """)
    else:
        st.success(f"✅ **Clearance:** This student is projected as **SAFE** to pass. (Risk Confidence: {1-prob:.2%})")
