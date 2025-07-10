import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
from PIL import Image
import base64

# --- Constants ---
MODEL_PATH = './Xception/Xception_fine_tuned.keras'
BACKGROUND_PATH = './Deployment/background.jpeg'
CLASS_NAMES = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']
TARGET_SIZE = (150, 150)

# --- Detailed Descriptions for Each Disease ---
DISEASE_EXPLANATIONS = {
    'CaS': (
        "🦷 **Caries Surface (CaS):**\n"
        "Dental caries on the tooth surface cause localized demineralization of enamel due to bacterial acids.\n"
        "If untreated, this can progress to dentin and pulp, leading to pain and infection.\n"
        "Early detection is crucial to avoid cavities and tooth loss.\n"
        "Common symptoms include discoloration, roughness, and sensitivity."
    ),
    'CoS': (
        "🦷 **Caries Occlusal Surface (CoS):**\n"
        "Decay affecting the chewing surfaces of molars and premolars, often in deep grooves.\n"
        "These areas are prone to trapping food and bacteria, leading to rapid cavity formation.\n"
        "Early treatment can prevent progression to severe tooth damage.\n"
        "Watch for dark spots, holes, or pain when chewing."
    ),
    'Gum': (
        "🪥 **Gum Disease (Periodontitis):**\n"
        "A serious gum infection that damages the soft tissue and destroys the bone supporting teeth.\n"
        "It is often caused by poor oral hygiene and plaque buildup.\n"
        "Symptoms include bleeding gums, bad breath, and loose teeth.\n"
        "Professional care is needed to prevent tooth loss and systemic effects."
    ),
    'MC': (
        "👄 **Mucosal Condition (MC):**\n"
        "Inflammation or changes in the lining of the mouth (oral mucosa).\n"
        "Causes can include trauma, infections, allergies, or systemic diseases.\n"
        "May present as redness, ulcers, swelling, or white patches.\n"
        "Monitoring and appropriate treatment can prevent complications."
    ),
    'OC': (
        "⚠️ **Oral Cancer (OC):**\n"
        "A malignant growth in the oral cavity that requires early detection for successful treatment.\n"
        "Risk factors include tobacco use, alcohol, and HPV infection.\n"
        "Common signs are non-healing ulcers, lumps, or persistent pain.\n"
        "Regular screening and biopsies are key to diagnosis and prevention."
    ),
    'OLP': (
        "📄 **Oral Lichen Planus (OLP):**\n"
        "A chronic inflammatory condition affecting the inner lining of the mouth.\n"
        "It may appear as white, lacy patches or painful sores.\n"
        "The cause is unknown but linked to immune system dysfunction.\n"
        "Management includes monitoring and symptom relief to prevent flare-ups."
    ),
    'OT': (
        "🔍 **Other Tooth Conditions (OT):**\n"
        "This category includes various non-specific dental issues like enamel defects or wear.\n"
        "Symptoms vary and may involve sensitivity, discoloration, or structural changes.\n"
        "A dental exam is recommended to identify the cause.\n"
        "Proper oral care and timely intervention can prevent worsening."
    )
}

# --- Custom CSS Styles ---
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
}
.main-container {
    background-color: rgba(0, 0, 0, 0.7);
    border-radius: 15px;
    padding: 30px;
    margin: 30px auto;
    max-width: 800px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
.results-box {
    background-color: rgba(30, 30, 30, 0.9);
    border-radius: 10px;
    padding: 20px;
    margin: 15px 0;
    border-left: 4px solid #4CAF50;
}
.title {
    color: white;
    text-align: center;
    font-size: 2.5rem;
    margin-bottom: 10px;
}
.subtitle {
    color: #4CAF50;
    text-align: center;
    font-size: 1.5rem;
    margin-bottom: 30px;
}
.result-text {
    color: white;
    font-size: 1.2rem;
    margin-bottom: 10px;
}
.highlight {
    color: #4CAF50;
    font-weight: bold;
}
.description {
    color: #CCCCCC;
    font-size: 1rem;
    margin-top: 10px;
    font-style: italic;
    white-space: pre-wrap;
}
.probability-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid #444;
}
.probability-row:last-child {
    border-bottom: none;
}
.probability-label {
    color: white;
    font-size: 1.1rem;
}
.probability-value {
    color: #4CAF50;
    font-size: 1.1rem;
}

/* Dark expander style */
div[data-testid="stExpander"] > details {
    background-color: rgba(30, 30, 30, 0.9);
    border: 1px solid #4CAF50;
    border-radius: 10px;
}
div[data-testid="stExpander"] summary {
    color: #4CAF50;
    font-weight: bold;
    font-size: 1.2rem;
}
div[data-testid="stExpander"] summary:hover {
    background-color: rgba(50, 50, 50, 0.9);
}
</style>
""", unsafe_allow_html=True)

# --- Page Setup ---
st.set_page_config(
    page_title="Dental Disease Classifier",
    page_icon="🦷",
    layout="centered"
)

# --- Background Image ---
def set_background(image_path):
    try:
        with open(image_path, "rb") as f:
            img_data = f.read()
        b64_encoded = base64.b64encode(img_data).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{b64_encoded}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.warning(f"Background image not found: {e}")

set_background(BACKGROUND_PATH)

# --- Load Model ---
@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        st.stop()
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_my_model()

# --- Main App ---
with st.container():
    st.markdown("""
    <div class="main-container">
        <h1 class="title">Dental Disease Classifier</h1>
        <h2 class="subtitle">AI-Powered Diagnosis System</h2>
    </div>
    """, unsafe_allow_html=True)

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Upload Tooth Image (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    key="file_uploader"
)

# --- Prediction Logic ---
if uploaded_file:
    with st.spinner('Analyzing...'):
        try:
            img = image.load_img(uploaded_file, target_size=TARGET_SIZE)
            img_array = image.img_to_array(img)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            predicted_class_idx = np.argmax(prediction, axis=1)[0]
            predicted_class = CLASS_NAMES[predicted_class_idx]
            confidence = np.max(prediction) * 100
            description = DISEASE_EXPLANATIONS.get(predicted_class, "No description available.")

            # Display Results
            with st.container():
                st.markdown("""
                <div class="main-container">
                    <h2 style="color: white; text-align: center;">Analysis Results</h2>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns([1, 1])
                with col1:
                    st.image(img, caption="Uploaded Image", width=250)

                with col2:
                    st.markdown(f"""
                    <div class="results-box">
                        <p class="result-text"><b>Prediction:</b> <span class="highlight">{predicted_class}</span></p>
                        <p class="result-text"><b>Confidence:</b> <span class="highlight">{confidence:.2f}%</span></p>
                        <p class="description">{description}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with st.expander("Detailed Probabilities", expanded=True):
                    for name, prob in zip(CLASS_NAMES, prediction[0]):
                        st.markdown(f"""
                        <div class="probability-row">
                            <span class="probability-label">{name}</span>
                            <span class="probability-value">{prob:.4f}</span>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error processing the image: {e}")
