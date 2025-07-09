#  Surveillance Insight Platform

A smart NLP-powered platform for detecting anomalies in system or security logs.  
Built using **Python**, **Flask**, **Logistic Regression**, and **TF-IDF** to classify textual events as `Normal` or `Anomaly`.

---

##  Features

-  **ML-based Anomaly Detection** using Logistic Regression
-  **REST API + Web Interface** (Flask backend)
-  **NLP Preprocessing** for input cleaning
-  Real-time classification of log messages
-  Simple UI for testing, and `/predict` API for automation
-  Deployable on **Render**, **Railway**, or any cloud host

---


##  ML Approach

- **Model**: Logistic Regression
- **Vectorizer**: TF-IDF
- **Input**: Cleaned log strings
- **Output**: `Anomaly` or `Normal`
- **Training Data**: Labeled log messages in `labeled_data.txt`

---

