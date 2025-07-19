# Fake News Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/Nagamanikanta/fake-news-detector)

This project is an end-to-end system for detecting fake news using a deep learning model. It classifies news articles as "Real" or "Fake" and exposes this functionality through a REST API built with FastAPI.

The core of the system is an LSTM (Long Short-Term Memory) model trained on a well-known news dataset. The model is optimized for deployment using TensorFlow Lite and includes multi-language support for English, Hindi, and Telugu.

---

## ✨ Features

* **Deep Learning Model:** Utilizes an LSTM network built with TensorFlow/Keras for sequence classification.
* **Optimized for Deployment:** The trained model is converted to the lightweight TensorFlow Lite (`.tflite`) format for efficient inference.
* **REST API:** A production-ready API built with FastAPI to serve predictions over HTTP.
* **Multi-language Support:** Classifies news in English (`en`), Hindi (`hi`), and Telugu (`te`) by translating input text to English before prediction.
* **Complete Workflow:** Demonstrates the full machine learning lifecycle from data preprocessing and training to deployment.

---

## 🛠️ Tech Stack

* **Backend:** Python, FastAPI
* **Machine Learning:** TensorFlow, Keras, Scikit-learn
* **Data Handling:** Pandas, NumPy
* **Translation:** Deep-Translator
* **API Server:** Uvicorn

---

## 📂 Project Structure

```
fake-news-detector/
│
├── data/                    # Raw dataset 
│   ├── Fake.csv
│   └── True.csv
│
├── processed_data/          # Processed NumPy arrays for training
│   ├── X_train.npy
│   └── ...
│
├── venv/                    # Virtual environment
│
├── main.py                  # FastAPI application script
├── model.tflite             # Optimized TFLite model for deployment
├── tokenizer.pickle         # Keras tokenizer for preprocessing text
│
├── preprocess.py            # Script for data cleaning and preprocessing
├── train.py                 # Script for training the LSTM model
├── convert_to_tflite.py     # Script to convert the Keras model to TFLite
├── test_api.py              # Script to test the running API
│
├── requirements.txt         # Project dependencies
├── REPORT.md                # Project documentation and results
└── README.md                
```
---

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

* Python 3.8+
* Git

### Installation

1.  **Clone the Repository**
    ```sh
    git clone [https://github.com/your-username/fake-news-detector.git](https://github.com/your-username/fake-news-detector.git)
    cd fake-news-detector
    ```

2.  **Create and Activate a Virtual Environment**
    ```sh
    # Create the environment
    python -m venv venv

    # Activate it
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Download the Dataset**
    The training data is not included in this repository.
    * Download the dataset from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).
    * Create a `data` folder in the project root.
    * Place `Fake.csv` and `True.csv` inside the `data` folder.

5.  **Reproduce the Model (Optional)**
    This repository includes the pre-trained `model.tflite` and `tokenizer.pickle`. To generate them yourself, run the scripts in order:
    ```sh
    # 1. Preprocess the raw data
    python preprocess.py

    # 2. Train the model and save it as .h5
    python train.py

    # 3. Convert the .h5 model to .tflite
    python convert_to_tflite.py
    ```

---

## 🏃‍♀️ Usage

### 1. Run the API Server

Start the FastAPI server using Uvicorn:
```sh
uvicorn main:app --reload
```
The API will be available at http://127.0.0.1:8000.

### 2. Test the API
You can interact with the API in several ways:

Interactive Docs: Open your browser and go to http://127.0.0.1:8000/docs to see the Swagger UI.

Using the Test Script: Run the test_api.py script to send sample requests for different languages.
```sh
python test_api.py
```

* Using cURL:
```sh
curl -X POST "[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)" \
-H "Content-Type: application/json" \
-d '{"text": "A new budget has been passed by the senate", "language": "en"}'
```
---

The expected output will be a JSON object:
```

{
  "prediction": "Real",
  "probability": "0.9987"
}

```
