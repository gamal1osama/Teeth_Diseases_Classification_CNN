# Teeth Diseases Classification using Convolutional Neural Networks

## Overview

This repository presents a comprehensive deep learning project focused on the classification of various teeth diseases using Convolutional Neural Networks (CNNs). The project explores different CNN architectures, including models built from scratch, as well as fine-tuned VGG16 and Xception models, to accurately identify and categorize dental conditions from image data. It also features a user-friendly web application for real-time predictions, built with Streamlit.

## Features

*   **Multiple CNN Architectures**: Implementation and comparison of:
    *   A custom-built CNN model from scratch.
    *   Fine-tuned VGG16 model for transfer learning.
    *   Fine-tuned Xception model for transfer learning.
*   **Comprehensive Data Preprocessing**: Techniques for preparing image datasets for optimal model training.
*   **Model Training and Evaluation**: Detailed notebooks showcasing the training process, evaluation metrics (including F1-score), and performance analysis for each model.
*   **Interactive Web Application**: A Streamlit-based application for easy deployment and real-time inference, allowing users to upload dental images and receive instant disease classifications.
*   **Research-Backed Approach**: Inclusion of relevant research papers and summarizations providing theoretical background and justification for the methodologies employed.

## Project Structure

The repository is organized into the following main directories:

*   `Building The Classifiers From Scratch/`: Contains the Jupyter Notebook for building and training a CNN model from scratch, along with related evaluation details and visualizations.
*   `Deployment/`: Houses the Streamlit web application (`app.py`) and its associated assets (`background.jpeg`) for deploying the trained models.
*   `Paper About The Problem/`: Includes research papers and their summaries that informed the project's approach to teeth disease classification.
*   `VGG16/`: Dedicated to the implementation, training, and evaluation of the VGG16-based CNN model, including its Jupyter Notebook and performance metrics.
*   `Xception/`: Contains the implementation, training, and evaluation of the Xception-based CNN model, including its Jupyter Notebook and performance metrics.
*   `requirements.txt`: Lists all necessary Python dependencies for setting up and running the project.

## Installation

To set up the project locally, follow these steps:

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/gamal1osama/Teeth_Diseases_Classification_CNN.git
    cd Teeth_Diseases_Classification_CNN
    ```

2.  **Create a virtual environment** (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training and Evaluation

To reproduce the training and evaluation of the models, navigate to the respective directories and open the Jupyter Notebooks:

*   For the custom CNN model: `Building The Classifiers From Scratch/teeth_diseases_classification_cnn.ipynb`
*   For the VGG16 model: `VGG16/teeth-diseases-classification-vgg16.ipynb`
*   For the Xception model: `Xception/teeth-diseases-classification-xception.ipynb`

Run all cells in the notebooks to train the models and view their performance metrics.

### Running the Web Application

To launch the Streamlit web application for real-time predictions:

1.  Ensure you are in the root directory of the cloned repository.
2.  Run the Streamlit application:

    ```bash
    streamlit run Deployment/app.py
    ```

    This will open the application in your web browser, where you can upload an image of teeth and get a classification result.

## Models and Performance

The project evaluates several CNN models for teeth disease classification. The key models and their F1-scores are:

*   **Custom CNN Model**: Achieved an F1-score of **98%**.
*   **VGG16 Fine-tuned Model**: Achieved an F1-score of **99%**.
*   **Xception Fine-tuned Model**: Achieved an F1-score of **99.42%**.

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.
