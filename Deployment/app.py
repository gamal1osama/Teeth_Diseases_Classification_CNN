###################  scaratch ################################
import numpy as np
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore


import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report # type: ignore
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # type: ignore

from PIL import Image


import os
import random


import warnings
warnings.filterwarnings("ignore")
#################################### VGG16 ###############################################

import numpy as np
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pandas as pd


import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import plot_model

from sklearn.metrics import classification_report # type: ignore
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # type: ignore


from PIL import Image


import os
import random


import warnings
warnings.filterwarnings("ignore")

######################################## Xception #########################

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore


import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import Xception
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


from PIL import Image


import os
import random


import warnings
warnings.filterwarnings("ignore")



############################################################################################
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

# --- Custom CSS Styles ---
st.markdown("""
<style>
/* Main background and containers */
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

/* Text styles */
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

/* Probability items */
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

/* Dark style for expander (open and closed) */
details summary {
    background-color: rgba(30, 30, 30, 0.9) !important;
    color: #4CAF50 !important;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
    font-size: 1.2rem;
    font-weight: bold;
    cursor: pointer;
}
details summary:hover {
    background-color: rgba(50, 50, 50, 0.9) !important;
}
details[open] > *:not(summary) {
    background-color: rgba(30, 30, 30, 0.9);
    border-radius: 10px;
    padding: 15px;
    margin-top: -10px;
    color: white;
}

/* Center and style uploader label */
section[data-testid="stFileUploader"] > label {
    display: block;
    text-align: center;
    font-size: 1.3rem;
    color: white;
    font-weight: bold;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# --- Page Setup ---
st.set_page_config(
    page_title="Dental Disease Classifier ",
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
        <h1 class="title">Dental Disease Classifier </h1>
        <h2 class="subtitle">AI-Powered Diagnosis System</h2>
    </div>
    """, unsafe_allow_html=True)

# --- File Uploader (Centered Label) ---
uploaded_file = st.file_uploader(
    "Upload Tooth Image (JPG, JPEG, PNG)",  # 👈 Label now styled and centered
    type=["jpg", "jpeg", "png"],
    key="file_uploader"
)

# --- Prediction Logic ---
if uploaded_file:
    with st.spinner('Analyzing...'):
        try:
            # Preprocess image
            img = image.load_img(uploaded_file, target_size=TARGET_SIZE)
            img_array = image.img_to_array(img)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction) * 100

            # Display Results
            with st.container():
                st.markdown("""
                <div class="main-container">
                    <h2 style="color: white; text-align: center;">Analysis Results</h2>
                """, unsafe_allow_html=True)

                # Image and Prediction in columns
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.image(img, caption="Uploaded Image", width=250)

                with col2:
                    st.markdown(f"""
                    <div class="results-box">
                        <p class="result-text"><b>Prediction:</b> <span class="highlight">{CLASS_NAMES[predicted_class]}</span></p>
                        <p class="result-text"><b>Confidence:</b> <span class="highlight">{confidence:.2f}%</span></p>
                    </div>
                    """, unsafe_allow_html=True)

                # Detailed Probabilities
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