# 🩺 Breast Cancer Detector App

This is a simple and interactive **Streamlit web app** that predicts whether a breast tumor is **Benign (non-cancerous)** or **Malignant (cancerous)** using a machine learning model trained on the Breast Cancer Wisconsin dataset.


## 📌 Overview

The app takes 30 input parameters related to characteristics of a breast cell nucleus — such as radius, texture, smoothness, etc. — and predicts the probability of cancer using a **Random Forest Classifier** from scikit-learn.


## 🚀 Features

- 🔢 Input 30 numerical diagnostic features
- 🧠 Machine Learning model trained on breast cancer dataset
- ✅ Predicts **Benign** or **Malignant**
- 📊 Shows probability/confidence of prediction
- 🧑‍💻 Developed in Python using **Streamlit**


## 📁 Dataset

- Source: `sklearn.datasets.load_breast_cancer()`
- Classes:  
  - `0` = Malignant  
  - `1` = Benign  
- 30 numerical features (mean, error, worst for each parameter)


## 🛠 Tech Stack

- **Python**
- **Streamlit**
- **Pandas**
- **scikit-learn**


## 💻 How to Run the App


# 1. Clone the repo
```bash
git clone https://github.com/your-username/breast-cancer-detector-app.git
cd breast-cancer-detector-app
```

# 2. Install required packages
```bash
pip install -r requirements.txt
```

# 3. Run the Streamlit app
```bash
streamlit run app.py
```



**_By Shriyash Shitole_**
