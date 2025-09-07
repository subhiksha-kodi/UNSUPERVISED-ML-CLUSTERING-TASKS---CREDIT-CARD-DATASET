# 📊 Credit Card Customer Clustering – Gradio App

## 🚀 Overview
This project provides **interactive customer segmentation** using clustering techniques via a Gradio web interface.  
It allows you to input customer credit card attributes and predict the cluster assignment using different clustering algorithms:  

- **KMeans**  
- **Hierarchical (Agglomerative) Clustering**  
- **DBSCAN**  

The dataset is based on credit card usage behavior and financial transactions.

---

## 🌐 Live Demo

- **Gradio (Hugging Face Space)**  
Interact online instantly:  
👉 [Credit Card Clustering Gradio App](https://huggingface.co/spaces/subhiksha-kodi/unsupervised-ml-clustering-tasks)

---

## 🔑 Login Credentials
- **Username:** `admin`  
- **Password:** `1234`  

> Only authenticated users can access the clustering interface.

---

## 📂 Dataset
The dataset (`clustered_data.csv`) contains the following columns:

- `BALANCE`  
- `BALANCE_FREQUENCY`  
- `PURCHASES`  
- `ONEOFF_PURCHASES`  
- `INSTALLMENTS_PURCHASES`  
- `CASH_ADVANCE`  
- `PURCHASES_FREQUENCY`  
- `ONEOFF_PURCHASES_FREQUENCY`  
- `PURCHASES_INSTALLMENTS_FREQUENCY`  
- `CASH_ADVANCE_FREQUENCY`  
- `CASH_ADVANCE_TRX`  
- `PURCHASES_TRX`  
- `CREDIT_LIMIT`  
- `PAYMENTS`  
- `MINIMUM_PAYMENTS`  
- `PRC_FULL_PAYMENT`  

⚠️ Columns like `CUST_ID` and `TENURE` are excluded since they don’t contribute meaningfully to clustering.

---

## ⚙️ Features
- 🔹 **Login Authentication**: Only users with correct credentials can access the app.  
- 🔹 **Choose Algorithm**: KMeans, Hierarchical, or DBSCAN.  
- 🔹 **Dynamic Input Form**: Enter values for 16 credit card usage features.  
- 🔹 **Cluster Prediction**: See which cluster the input profile belongs to.  
- 🔹 **Outlier Detection**: DBSCAN identifies whether a profile is an outlier (`-1`).  
- 🔹 **Robust Scaling**: Features are scaled with `RobustScaler` to handle skewness and outliers.  

---

## 🛠️ Installation
Clone the repo and install dependencies:

```bash
git clone https://huggingface.co/spaces/subhiksha-kodi/unsupervised-ml-clustering-tasks
cd unsupervised-ml-clustering-tasks
pip install -r requirements.txt
