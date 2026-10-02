import pandas as pd
import joblib
import streamlit as st

# PAGE CONFIG

st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM THEME

st.markdown(
    """
    <style>
    :root {
        --page-bg: #0B1B2B;
        --sidebar-bg: #071522;
        --card-bg: #0F2234;
        --heading: #EAF2F8;
        --body-text: #B8C7D6;
        --muted-text: #8297AA;
        --accent: #21C7B7;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 82% 10%,
                rgba(33, 199, 183, 0.055),
                transparent 30%
            ),
            radial-gradient(
                circle at 15% 85%,
                rgba(55, 115, 170, 0.065),
                transparent 32%
            ),
            var(--page-bg);
        color: var(--body-text);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #071522 0%,
            #06121E 100%
        );
        border-right: 1px solid rgba(148, 170, 190, 0.08);
    }

    [data-testid="stSidebar"] * {
        color: var(--body-text);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--heading) !important;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 27, 43, 0.94);
    }

    h1, h2, h3 {
        color: var(--heading) !important;
        letter-spacing: -0.02em;
    }

    p, label, .stMarkdown {
        color: var(--body-text);
    }

    .stCaption {
        color: var(--muted-text) !important;
    }

    [data-testid="stMarkdownContainer"] strong {
        color: var(--heading);
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(15, 34, 52, 0.58);
        border-color: rgba(148, 170, 190, 0.11);
        animation: cardEnter 0.55s ease-out both;
        transition:
            transform 0.28s ease,
            border-color 0.28s ease,
            box-shadow 0.28s ease,
            background 0.28s ease;
    }

    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-5px);
        background: rgba(18, 42, 61, 0.78);
        border-color: rgba(33, 199, 183, 0.42);
        box-shadow:
            0 12px 30px rgba(0, 0, 0, 0.30),
            0 0 18px rgba(33, 199, 183, 0.10);
    }

    .stButton > button {
        transition:
            transform 0.22s ease,
            box-shadow 0.22s ease,
            background 0.22s ease;
    }

    .stButton > button:hover {
        transform: translateY(-3px) scale(1.01);
        box-shadow:
            0 9px 24px rgba(0, 0, 0, 0.28),
            0 0 20px rgba(33, 199, 183, 0.22);
    }

    .stButton > button:active {
        transform: translateY(-1px) scale(0.99);
    }

    .stNumberInput input,
    [data-baseweb="select"] > div {
        transition:
            border-color 0.2s ease,
            box-shadow 0.2s ease,
            transform 0.2s ease;
    }

    .stNumberInput input:hover,
    [data-baseweb="select"] > div:hover {
        border-color: rgba(33, 199, 183, 0.45) !important;
        box-shadow: 0 0 0 2px rgba(33, 199, 183, 0.08);
    }

    .stNumberInput input:focus,
    [data-baseweb="select"] > div:focus-within {
        border-color: rgba(33, 199, 183, 0.65) !important;
        box-shadow: 0 0 0 3px rgba(33, 199, 183, 0.12);
    }

    [data-testid="stWidgetLabel"] p {
        color: #AFC0D0 !important;
        font-weight: 500;
    }

    .stNumberInput input,
    [data-baseweb="select"] * {
        color: #172331 !important;
    }

    .stNumberInput input,
    [data-baseweb="select"] > div {
        background: #F1F4F7 !important;
    }

    button {
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(33, 199, 183, 0.16);
    }

    @keyframes cardEnter {
        from {
            opacity: 0;
            transform: translateY(8px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @media (prefers-reduced-motion: reduce) {
        [data-testid="stVerticalBlockBorderWrapper"] {
            animation: none;
            transition: none;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# LOAD MODEL + DATA

model = joblib.load("heart_disease_model.pkl")
df = pd.read_csv("heart.csv")

feature_columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal"
]

# SIDEBAR

with st.sidebar:

    st.subheader("System")

    with st.container(border=True):
        st.write("Random Forest Model")
        st.caption(
            "Machine learning classification model for heart disease prediction."
        )

    with st.container(border=True):
        st.write("13 Clinical Features")
        st.caption(
            "Patient demographics, clinical measurements and diagnostic indicators."
        )

    st.subheader("Technology")

    with st.container(border=True):
        st.caption("Python · Streamlit")
        st.caption("ML · Random Forest")
        st.caption("Data · Pandas")

    st.subheader("Important")
    st.warning(
        "This application is an educational machine learning project "
        "and should not be used as a substitute for professional medical advice."
    )

# HERO

st.title("Heart Disease Prediction")

st.caption(
    "Analyze clinical patient information using a machine learning "
    "classification model and generate an AI-assisted prediction."
)

st.info(
    "Random Forest | 13 Clinical Features | AI Prediction | Healthcare Analytics"
)

# PATIENT INFORMATION

st.caption("01 · PATIENT PROFILE")
st.header("Patient Information")
st.write("Enter the patient's basic demographic information.")

with st.container(border=True):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=45
        )

    with col2:
        sex_option = st.selectbox(
            "Sex",
            ["Female", "Male"]
        )

    sex = 0 if sex_option == "Female" else 1

# CLINICAL INFORMATION

st.caption("02 · CLINICAL DATA")
st.header("Clinical Information")

st.write(
    "Provide the patient's cardiovascular measurements and clinical indicators."
)

with st.container(border=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        cp_option = st.selectbox(
            "Chest Pain Type",
            [
                "Typical Angina",
                "Atypical Angina",
                "Non-anginal Pain",
                "Asymptomatic"
            ]
        )

        cp = {
            "Typical Angina": 0,
            "Atypical Angina": 1,
            "Non-anginal Pain": 2,
            "Asymptomatic": 3
        }[cp_option]

        trestbps = st.number_input(
            "Resting Blood Pressure",
            min_value=50,
            max_value=250,
            value=120
        )

    with col2:
        chol = st.number_input(
            "Cholesterol",
            min_value=50,
            max_value=700,
            value=200
        )

        fbs_option = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            ["No", "Yes"]
        )

        fbs = 0 if fbs_option == "No" else 1

    with col3:
        restecg_option = st.selectbox(
            "Resting ECG",
            [
                "Normal",
                "ST-T Wave Abnormality",
                "Left Ventricular Hypertrophy"
            ]
        )

        restecg = {
            "Normal": 0,
            "ST-T Wave Abnormality": 1,
            "Left Ventricular Hypertrophy": 2
        }[restecg_option]

        thalach = st.number_input(
            "Maximum Heart Rate",
            min_value=50,
            max_value=250,
            value=150
        )

# ADDITIONAL TEST INFORMATION

st.caption("03 · DIAGNOSTIC INDICATORS")
st.header("Additional Test Information")

st.write(
    "Complete the remaining diagnostic parameters used by the model."
)

with st.container(border=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        exang_option = st.selectbox(
            "Exercise-Induced Angina",
            ["No", "Yes"]
        )

        exang = 0 if exang_option == "No" else 1

        oldpeak = st.number_input(
            "ST Depression (Oldpeak)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1
        )

    with col2:
        slope_option = st.selectbox(
            "ST Segment Slope",
            [
                "Upsloping",
                "Flat",
                "Downsloping"
            ]
        )

        slope = {
            "Upsloping": 0,
            "Flat": 1,
            "Downsloping": 2
        }[slope_option]

        ca_values = df["ca"].unique().tolist()

        ca = st.selectbox(
            "Major Vessels",
            ca_values
        )

    with col3:
        thal_option = st.selectbox(
            "Thalassemia",
            [
                "Normal",
                "Fixed Defect",
                "Reversible Defect",
                "Other / Unknown"
            ]
        )

        thal = {
            "Normal": 0,
            "Fixed Defect": 1,
            "Reversible Defect": 2,
            "Other / Unknown": 3
        }[thal_option]

# INPUT PREVIEW

st.caption("04 · INPUT REVIEW")
st.header("Patient Data Preview")

st.write(
    "Review the key clinical values before running the prediction."
)

with st.container(border=True):
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.metric("Age", age)

    with c2:
        st.metric("Blood Pressure", trestbps)

    with c3:
        st.metric("Cholesterol", chol)

    with c4:
        st.metric("Max Heart Rate", thalach)

    with c5:
        st.metric("Major Vessels", ca)

    with c6:
        st.metric("Oldpeak", oldpeak)

    # Patient data graph - updates when inputs change
    st.subheader("Patient Data Graph")

    patient_graph = pd.DataFrame({
        "Parameter": [
            "Age",
            "Blood Pressure",
            "Cholesterol",
            "Max Heart Rate",
            "Major Vessels",
            "Oldpeak"
        ],
        "Value": [
            age,
            trestbps,
            chol,
            thalach,
            ca,
            oldpeak
        ]
    }).set_index("Parameter")

    st.bar_chart(
        patient_graph,
        use_container_width=True
    )

# FEATURE IMPORTANCE - ADDED SECTION

st.caption("05 · MODEL EXPLAINABILITY")
st.header("Feature Importance")

st.write(
    "This chart shows which clinical features the Random Forest model "
    "uses most when making predictions. Higher importance means the "
    "feature contributes more to the model's decision process."
)

if hasattr(model, "feature_importances_"):

    feature_names = {
        "age": "Age",
        "sex": "Sex",
        "cp": "Chest Pain",
        "trestbps": "Blood Pressure",
        "chol": "Cholesterol",
        "fbs": "Fasting Blood Sugar",
        "restecg": "Resting ECG",
        "thalach": "Maximum Heart Rate",
        "exang": "Exercise-Induced Angina",
        "oldpeak": "Oldpeak",
        "slope": "ST Segment Slope",
        "ca": "Major Vessels",
        "thal": "Thalassemia"
    }

    importance_df = pd.DataFrame({
        "Feature": [
            feature_names.get(feature, feature)
            for feature in feature_columns
        ],
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    with st.container(border=True):
        st.subheader("Top Model Features")

        top_features = importance_df.head(5).copy()

        top_features["Importance (%)"] = (
            top_features["Importance"] * 100
        ).round(2)

        st.dataframe(
            top_features[["Feature", "Importance (%)"]],
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Feature Importance Chart")

        importance_chart = importance_df.set_index("Feature")

        st.bar_chart(
            importance_chart,
            use_container_width=True
        )

        st.caption(
            "Feature importance values are calculated from the trained "
            "Random Forest model. They describe model-level feature "
            "importance, not a patient's individual medical risk."
        )

else:
    st.info(
        "Feature importance is not available for this model."
    )

# PREDICTION BUTTON

st.write("")

predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])

with predict_col2:
    predict_button = st.button(
        "RUN AI PREDICTION",
        use_container_width=True
    )

# ML LOGIC

if predict_button:

    input_data = pd.DataFrame(
        [{
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }],
        columns=feature_columns
    )

    prediction = model.predict(input_data)[0]

    # Get model probabilities if supported
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]

        no_disease_probability = probabilities[0] * 100
        disease_probability = probabilities[1] * 100
    else:
        no_disease_probability = None
        disease_probability = None

    # PATIENT RISK SCORE

    if disease_probability is not None:
        risk_score = disease_probability

        if risk_score < 30:
            risk_level = "Low Risk Indicator"
        elif risk_score < 60:
            risk_level = "Moderate Risk Indicator"
        else:
            risk_level = "Higher Risk Indicator"

    # RESULT
    
    st.caption("06 · AI ANALYSIS")
    st.header("Prediction Result")

    st.write(
        "Machine learning analysis based on the submitted clinical data."
    )

    with st.container(border=True):
        if prediction == 1:
            st.error("Heart Disease Detected")
        else:
            st.success("No Heart Disease Detected")

    # PATIENT RISK SCORE

    if disease_probability is not None:
        st.subheader("Patient Risk Score")

        st.caption(
            "Model-derived indicator based on the prediction probability; "
            "it is not a clinical diagnosis or validated medical risk score."
        )

        r1, r2 = st.columns([1, 2])

        with r1:
            st.metric(
                "Risk Score",
                f"{risk_score:.1f}%"
            )

        with r2:
            st.write(f"**{risk_level}**")
            st.progress(
                min(100, max(0, int(risk_score)))
            )

    # PREDICTION PROBABILITY VISUALIZATION

    if no_disease_probability is not None:

        st.subheader("Prediction Confidence")

        st.caption(
            "The percentages below represent the model's predicted "
            "probabilities for each class."
        )

        p1, p2 = st.columns(2)

        with p1:
            st.metric(
                label="No Heart Disease",
                value=f"{no_disease_probability:.2f}%"
            )

            st.progress(
                min(100, max(0, int(no_disease_probability)))
            )

        with p2:
            st.metric(
                label="Heart Disease",
                value=f"{disease_probability:.2f}%"
            )

            st.progress(
                min(100, max(0, int(disease_probability)))
            )

    else:
        st.info(
            "This model does not provide predict_proba(), "
            "so probability visualization is unavailable."
        )

    # PATIENT INPUT EXPLANATION
    
    st.subheader("Patient Data Explanation")

    explanation = []

    # Cholesterol explanation
    if chol < 200:
        cholesterol_status = "Below 200 mg/dL"
    elif chol < 240:
        cholesterol_status = "200–239 mg/dL"
    else:
        cholesterol_status = "240 mg/dL or higher"

    explanation.append({
        "Parameter": "Cholesterol",
        "Entered Value": f"{chol} mg/dL",
        "Input Explanation": cholesterol_status
    })

    # Blood pressure explanation
    if trestbps < 120:
        bp_status = "Below 120 mmHg"
    elif trestbps < 130:
        bp_status = "120–129 mmHg"
    else:
        bp_status = "130 mmHg or higher"

    explanation.append({
        "Parameter": "Resting Blood Pressure",
        "Entered Value": f"{trestbps} mmHg",
        "Input Explanation": bp_status
    })

    # Maximum heart rate
    explanation.append({
        "Parameter": "Maximum Heart Rate",
        "Entered Value": f"{thalach} bpm",
        "Input Explanation": "Value entered for model prediction"
    })

    # Chest pain
    explanation.append({
        "Parameter": "Chest Pain Type",
        "Entered Value": cp_option,
        "Input Explanation": "Selected patient input"
    })

    # Exercise-induced angina
    explanation.append({
        "Parameter": "Exercise-Induced Angina",
        "Entered Value": exang_option,
        "Input Explanation": "Selected patient input"
    })

    # Oldpeak
    explanation.append({
        "Parameter": "Oldpeak",
        "Entered Value": oldpeak,
        "Input Explanation": "ST depression value entered for model"
    })

    explanation_df = pd.DataFrame(explanation)

    st.dataframe(
        explanation_df,
        use_container_width=True,
        hide_index=True
    )

    # PATIENT SUMMARY

    st.subheader("Patient Summary")

    summary = pd.DataFrame({
        "Parameter": [
            "Age",
            "Sex",
            "Chest Pain",
            "Resting BP",
            "Cholesterol",
            "Maximum Heart Rate",
            "Exercise Angina",
            "Oldpeak",
            "Major Vessels",
            "Thalassemia"
        ],
        "Value": [
            age,
            sex_option,
            cp_option,
            trestbps,
            chol,
            thalach,
            exang_option,
            oldpeak,
            ca,
            thal_option
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

# MODEL INFORMATION

with st.expander("Model Information"):
    st.write(
        "This application uses a Random Forest classification model "
        "trained on heart disease clinical data."
    )

    st.write(
        "The model receives 13 clinical features and predicts whether "
        "heart disease is detected."
    )

# FOOTER

st.caption(
    "HeartAI · AI Healthcare Prediction System · "
    "Educational Machine Learning Project · "
    "Not a substitute for professional medical advice"
)