# Ethereum Transaction Fraud Detection System

This project is a Cyber Security AI/ML Proof of Concept (POC) designed to detect fraudulent Ethereum transactions using Machine Learning. It analyzes transaction patterns, gas usage, and account behavior to classify activities as either "Legit" or "Fraudulent."

## 🚀 Features

* **Data Analysis:** Extensive Exploratory Data Analysis (EDA) on Ethereum transaction datasets.
* **Preprocessing:** Automated handling of missing values, categorical mapping, and feature selection.
* **Fraud Detection Model:** Implements a **Random Forest Classifier** to predict fraud with high accuracy.
* **Simulation:** Includes a module to generate synthetic transaction data and simulate real-time fraud detection.
* **Performance Metrics:** Evaluates the model using Confusion Matrix, Precision, Recall, and F1-Score.

## 🛠️ Tech Stack

* **Python 3.x**
* **Pandas & NumPy:** Data manipulation and numerical operations.
* **Scikit-Learn:** Machine learning model training and evaluation.
* **Matplotlib & Seaborn:** Data visualization.
* **Joblib:** Model persistence (saving/loading trained models).

## 📂 Project Structure

* `Cyber AIML POC.ipynb`: The main Jupyter Notebook containing the end-to-end pipeline (Data loading -> Cleaning -> Training -> Evaluation).
* `transaction_dataset.csv`: (Required) The raw dataset containing Ethereum transaction history.
* `fraud_detection_model.pkl`: The saved trained model file for quick deployment.
* `sample_new_transactions.csv`: Generated sample data for testing predictions.

## 📊 Dataset Details

The model is trained on a dataset containing 50+ features, including:
* **Transaction Frequency:** Avg time between sent/received transactions.
* **Value Stats:** Min/Max/Avg values sent and received.
* **Gas/Contract Info:** Total Ether sent to contracts, ERC20 token transfers.
* **Flag:** Target variable (0 = Legit, 1 = Fraud).

## ⚙️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/PrathamMJain147/Ethereum-Fraud-Detection.git](https://github.com/PrathamMJain147/Ethereum-Fraud-Detection.git)
    cd Ethereum-Fraud-Detection
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn joblib
    ```

3.  **Run the analysis:**
    Open `Cyber AIML POC.ipynb` in Jupyter Notebook or VS Code to see the training process and visualizations.

4.  **Use the trained model:**
    The notebook includes a script to load `fraud_detection_model.pkl` and predict new transactions:
    ```python
    import joblib
    model = joblib.load("fraud_detection_model.pkl")
    prediction = model.predict(new_data)
    ```

## 📈 Model Performance

The Random Forest model achieved excellent results on the test set:
* **Accuracy:** ~100%
* **Precision (Fraud):** 1.00
* **Recall (Fraud):** 1.00

*(Note: High accuracy suggests the dataset contains distinct patterns separating fraudulent bots/accounts from legitimate users.)*


---
*Developed as a Cyber Security AIML Proof of Concept.*
