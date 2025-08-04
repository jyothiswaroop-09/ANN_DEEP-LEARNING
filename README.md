<h1>DEEP LEARNING PROJECT</h1>

# 🤖 Customer Churn Prediction using Deep Learning (TensorFlow ANN)

A deep learning project to predict customer churn using an Artificial Neural Network (ANN) built with TensorFlow/Keras.

---

## 🚀 Project Overview

- **Goal**: Predict whether a customer will churn using structured bank customer data.
- **Approach**: Binary classification using a deep neural network.
- **Framework**: TensorFlow + Keras
- **Dataset**: Churn_Modelling.csv

---

## 🧠 Model Architecture (ANN)

- Input Layer: Preprocessed numerical/categorical features
- Hidden Layers: 2–3 Dense layers with ReLU activation
- Output Layer: Sigmoid activation for binary classification
- Loss: `binary_crossentropy`
- Optimizer: `adam`
- Metrics: Accuracy, Precision, Recall

---

## 📁 Project Structure
```
churn-tf-ann/
├── data/ # Raw dataset
├── models/ # Saved models (.h5)
├── notebooks/ # Jupyter notebooks (EDA, training)
│ └──experiments
│ └──hyperparametertuning ann
├── pickling Files/
│ └── label_encoder_gender.pkl
│ └──onehot_encoder_geo.pkl
│ └──scaler.pkl
├── app/ # Streamlit frontend
│ └── app.py
├── requirements.txt
├── README.md
└── main.py
```


---

## 📊 Features Used

- Credit Score  
- Geography (One-hot encoded)  
- Gender (Label encoded)  
- Age  
- Tenure  
- Balance  
- Num of Products  
- Has Credit Card  
- Is Active Member  
- Estimated Salary  

Target: **Exited** (1 = Churn, 0 = Stay)

---

## 🧪 Preprocessing Steps

- Dropped: `RowNumber`, `CustomerId`, `Surname`
- Label Encoding: `Gender`
- One-Hot Encoding: `Geography`
- Feature Scaling: `StandardScaler`
- Class Balancing: `SMOTE` on training set

---

## ⚙️ Model Training Workflow

### 1️⃣ Clone and Setup

```bash
git clone https://github.com/jyothiswaroop-09/ANN_DEEP-LEARNING.git
cd churn-tf-ann
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

2️⃣ Run Preprocessing + Training
bash
Copy code
python main.py

This runs:
Data preprocessing
Train/validation split
ANN model training using TensorFlow/Keras
Evaluation and saving best model

3️⃣ Launch Streamlit App
bash
Copy code
streamlit run app/app.py

🧠 Model Evaluation
Accuracy
Precision / Recall
F1 Score
ROC-AUC Curve
Confusion Matrix

🖥 Sample UI Output (Streamlit)
Predicts churn probability and shows clear interpretation.

Input fields for customer profile.

Displays:

"Customer is likely to churn" ✅

"Customer is not likely to churn" ❌

📦 Requirements
Main packages:
text
Copy code
tensorflow
scikit-learn
pandas
numpy
matplotlib
streamlit
imbalanced-learn
Full list in requirements.txt.

📌 Future Enhancements
Integrate EarlyStopping & ModelCheckpoint

👨‍💻 Author
Jyothi Swaroop
GitHub: jyothiswaroop-09
Email: swaroop.motupalli@gmail.com


