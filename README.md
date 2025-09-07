# 📊 Credit Card Customer Clustering – Streamlit App

## 🚀 Overview
This project is a **Streamlit web application** for interactive customer segmentation using clustering techniques.  
It allows you to input customer credit card attributes and predict the cluster assignment using different clustering algorithms:  

- **KMeans**  
- **Hierarchical (Agglomerative) Clustering**  
- **DBSCAN**  

The dataset is based on credit card usage behavior and financial transactions.

---

## 🌐 Live Demo
You can try the app here:  
👉 [Credit Card Clustering Streamlit App](https://unsupervised-ml-clustering-tasks---credit-card-dataset-vstnmbq.streamlit.app/)

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
- 🔹 **Choose Algorithm**: KMeans, Hierarchical, or DBSCAN.  
- 🔹 **Dynamic Input Form**: Enter values for 16 credit card usage features.  
- 🔹 **Cluster Prediction**: See which cluster the input profile belongs to.  
- 🔹 **Outlier Detection**: DBSCAN identifies whether a profile is an outlier (`-1`).  
- 🔹 **Robust Scaling**: Features are scaled with `RobustScaler` to handle skewness and outliers.  

---

## 🛠️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/credit-card-clustering.git
cd credit-card-clustering
pip install -r requirements.txt
```

**requirements.txt**
```
streamlit
pandas
numpy
scikit-learn
```

---

## ▶️ Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Then open the provided local URL in your browser (default: http://localhost:8501).

---

## 📊 Example Workflow
1. Select an algorithm (e.g., **KMeans**).  
2. Provide values for customer features (e.g., balance, purchases, payments).  
3. Click **Find Cluster**.  
4. The app displays the cluster ID or marks it as an **outlier** (DBSCAN).  

---

## 🧠 Future Improvements
- Add **PCA/TSNE visualization** of clusters.  
- Export clustered dataset as **CSV download**.  
- Add **silhouette score & metrics** to evaluate clustering quality.  

---

## 👩‍💻 Author
Developed by **Subhiksha** ✨  
