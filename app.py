import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

st.title("🎓 Student Academic Performance Predictor")
st.write("Enter the student's current metrics below to check their academic risk status.")

# Cache the data loading and training process so it only runs once when the app starts
@st.cache_resource
def train_model_live():
    # 1. Fetch data directly from the reliable mirror
    url = "https://raw.githubusercontent.com/KunjalJethwani/StudentPerformance/master/student-por.csv"
    df = pd.read_csv(url, sep=';')
    
    # 2. Target Engineering
    df['at_risk'] = np.where(df['G3'] < 10, 1, 0)
    df = df.drop(columns=['G3'])
    
    X = df.drop(columns=['at_risk'])
    y = df['at_risk']
    
    # 3. Separate feature types
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # 4. Preprocessing setup
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
        ])
    
    # 5. Fit pipeline and train the best model architecture found in Colab
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_proc = preprocessor.fit_transform(X_train)
    
    # Re-use our tuned optimal hyperparameters from the grid search
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
    model.fit(X_train_proc, y_train)
    
    return preprocessor, model, X.columns.tolist(), num_cols

# Run the live training
preprocessor, model, features, num_cols = train_model_live()

st.subheader("Student Metrics Input")

# Create user interface inputs
g1 = st.slider("G1 Grade (First Term: 0 to 20)", 0, 20, 10)
g2 = st.slider("G2 Grade (Second Term: 0 to 20)", 0, 20, 10)
absences = st.slider("Total Absences", 0, 93, 4)
failures = st.selectbox("Number of Past Class Failures", [0, 1, 2, 3, 4])
studytime = st.slider("Weekly Study Time Level (1: <2 hrs, 2: 2-5 hrs, 3: 5-10 hrs, 4: >10 hrs)", 1, 4, 2)

# Build custom input matrix matching training schema
input_dict = {col: [0] if col in num_cols else ['M'] for col in features}
input_dict['G1'] = [g1]
input_dict['G2'] = [g2]
input_dict['absences'] = [absences]
input_dict['failures'] = [failures]
input_dict['studytime'] = [studytime]

input_df = pd.DataFrame(input_dict)

if st.button("Predict Risk Status"):
    # Apply transformation matrix and predict
    processed_input = preprocessor.transform(input_df)
    pred = model.predict(processed_input)[0]
    prob = model.predict_proba(processed_input)[0][1]
    
    st.write("---")
    if pred == 1:
        st.error(f"Prediction: ⚠️ AT-RISK of failing the course. (Failure Probability: {prob:.2%})")
    else:
        st.success(f"Prediction: ✅ SAFE. Projected to pass. (Failure Probability: {prob:.2%})")
