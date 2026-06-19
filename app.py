import streamlit as st
import pandas as pd
import joblib

st.title("🎓 Student Academic Performance Predictor")
st.write("Enter the student's current metrics below to check their academic risk status.")

@st.cache_resource
def load_saved_pipeline():
    return joblib.load('student_model_pipeline.pkl')

saved_data = load_saved_pipeline()
preprocessor = saved_data['preprocessor']
model = saved_data['model']
features = saved_data['features']

st.subheader("Student Metrics Input")

g1 = st.slider("G1 Grade (First Term: 0 to 20)", 0, 20, 10)
g2 = st.slider("G2 Grade (Second Term: 0 to 20)", 0, 20, 10)
absences = st.slider("Total Absences", 0, 93, 4)
failures = st.selectbox("Number of Past Class Failures", [0, 1, 2, 3, 4])
studytime = st.slider("Weekly Study Time Level (1: <2 hrs, 2: 2-5 hrs, 3: 5-10 hrs, 4: >10 hrs)", 1, 4, 2)

input_dict = {col: [0] if col in saved_data['num_cols'] else ['M'] for col in features}

# Replace the specific variables we want to predict with
input_dict['G1'] = [g1]
input_dict['G2'] = [g2]
input_dict['absences'] = [absences]
input_dict['failures'] = [failures]
input_dict['studytime'] = [studytime]

input_df = pd.DataFrame(input_dict)

# Prediction button execution
if st.button("Predict Risk Status"):
    processed_input = preprocessor.transform(input_df)
    pred = model.predict(processed_input)[0]
    prob = model.predict_proba(processed_input)[0][1]
    
    st.write("---")
    if pred == 1:
        st.error(f"Prediction: ⚠️ AT-RISK of failing the course. (Failure Probability: {prob:.2%})")
    else:
        st.success(f"Prediction: ✅ SAFE. Projected to pass. (Failure Probability: {prob:.2%})")
