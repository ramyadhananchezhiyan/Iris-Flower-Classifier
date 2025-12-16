import streamlit as st
import numpy as np
import joblib

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌸",
    layout="centered"
)

# ---------------------------
# Custom CSS for aqua + white theme
# ---------------------------
st.markdown(
    """
    <style>
    /* Background */
    .stApp {
        background: linear-gradient(135deg, #e8ffff, #ffffff);
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main container */
    .main-card {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2rem 2.5rem;
        border-radius: 1.5rem;
        box-shadow: 0 18px 40px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 180, 200, 0.15);
    }

    /* Title */
    .title-text {
        font-size: 2rem;
        font-weight: 700;
        color: #008c9e;
        text-align: center;
        margin-bottom: 0.3rem;
    }

    .subtitle-text {
        font-size: 0.95rem;
        color: #3b6978;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Labels */
    label, .stSlider label {
        color: #22556b !important;
        font-weight: 500;
        font-size: 0.9rem;
    }

    /* Button */
    .stButton>button {
        background: linear-gradient(90deg, #00bcd4, #00acc1);
        color: white;
        border-radius: 999px;
        border: none;
        padding: 0.6rem 1.8rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        box-shadow: 0 8px 20px rgba(0, 172, 193, 0.35);
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 28px rgba(0, 172, 193, 0.45);
        cursor: pointer;
    }

    /* Prediction box */
    .prediction-box {
        margin-top: 1.5rem;
        padding: 1.2rem 1.4rem;
        border-radius: 1.2rem;
        background: linear-gradient(135deg, #e0f7fa, #ffffff);
        border: 1px solid rgba(0, 188, 212, 0.4);
    }
    .prediction-label {
        font-size: 0.9rem;
        color: #3b6978;
        margin-bottom: 0.3rem;
    }
    .prediction-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #006064;
    }
    .prediction-prob {
        font-size: 0.85rem;
        color: #4f7a8a;
        margin-top: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Load model and metadata
# ---------------------------
@st.cache_resource
def load_model():
    model = joblib.load("iris_model.pkl")
    # If you want to be extra-safe, you can store class names here:
    class_names = ["Setosa", "Versicolor", "Virginica"]
    return model, class_names

model, class_names = load_model()

# ---------------------------
# UI Layout
# ---------------------------
st.markdown('<div class="main-card title-text">Iris Flower Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-text">Adjust the measurements below and let the model predict the Iris species 💧</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider(
        "Sepal length (cm)",
        min_value=4.0, max_value=8.0, value=5.8, step=0.1
    )
    sepal_width = st.slider(
        "Sepal width (cm)",
        min_value=2.0, max_value=4.5, value=3.0, step=0.1
    )

with col2:
    petal_length = st.slider(
        "Petal length (cm)",
        min_value=1.0, max_value=7.0, value=4.35, step=0.1
    )
    petal_width = st.slider(
        "Petal width (cm)",
        min_value=0.1, max_value=2.5, value=1.3, step=0.1
    )

# Prepare feature vector in the SAME ORDER as training
features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Predict button
if st.button("Predict species"):
    pred_class_index = model.predict(features)[0]
    pred_class_index = int(np.around(pred_class_index))
    species_name = class_names[pred_class_index]

    st.markdown(
        f"""
        <div class="prediction-box">
            <div class="prediction-label">Predicted species</div>
            <div class="prediction-value">{species_name}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)
