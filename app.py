st.markdown("---")
    st.write("📊 **Enter Available Coursework Marks (0 - 20 Marks per Assignment)**")
    
    # Define real, practical university assessment blueprints for each module
    syllabus_blueprints = {
        "COM 763: Advanced Machine Learning": [
            "Coursework 1: Exploratory Data Analysis Portfolio",
            "Coursework 2: Model Training & Hyperparameter GridSearch",
            "Coursework 3: Streamlit Deployment & Technical Report",
            "Final Examination"
        ],
        "COM 742: Enterprise Data Systems": [
            "Coursework 1: Relational Schema & SQL Design",
            "Coursework 2: NoSQL Database Scaling Lab",
            "Coursework 3: Distributed Data Infrastructure Project",
            "Final Examination"
        ],
        "COM 711: Software Engineering Foundations": [
            "Coursework 1: Requirements Specification & UML Diagrams",
            "Coursework 2: Object-Oriented Programming Core Build",
            "Final Practical Code Defense",
            "Final Examination"
        ],
        "COM 705: Artificial Intelligence Principles": [
            "Coursework 1: Search Algorithm Optimization Lab",
            "Coursework 2: Neural Network Implementation Project",
            "Final Examination"
        ]
    }
    
    # Fetch the assessment names for whatever module the user selected
    active_syllabus = syllabus_blueprints[selected_module]
    
    # Dynamically generate sliders using the real assignment titles instead of generic numbers
    grades = []
    for i in range(completed_assignments):
        # Use the real syllabus name from our blueprint list
        assignment_label = active_syllabus[i]
        score = st.slider(f"✨ {assignment_label}", min_value=0, max_value=20, value=10)
        grades.append(score)
