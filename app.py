import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import RobustScaler

# ---------------------------
# Load dataset
# ---------------------------
data = pd.read_csv("clustered_data.csv")

# Final selected features for clustering
selected_features = [
    "BALANCE",
    "BALANCE_FREQUENCY",
    "PURCHASES",
    "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE",
    "PURCHASES_FREQUENCY",
    "ONEOFF_PURCHASES_FREQUENCY",
    "PURCHASES_INSTALLMENTS_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY",
    "CASH_ADVANCE_TRX",
    "PURCHASES_TRX",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "MINIMUM_PAYMENTS",
    "PRC_FULL_PAYMENT"
]

X = data[selected_features].values

# Scale features with RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="🔍 Credit Card Clustering App", layout="centered")
st.title("🔍 Customer Clustering App")

st.markdown("Choose algorithm, provide inputs, and find the cluster for a customer profile.")

# Algorithm selection
algorithm = st.selectbox("Select Clustering Algorithm", ["KMeans", "Hierarchical", "DBSCAN"])

# Extra input for KMeans
k = None
if algorithm == "KMeans":
    k = st.number_input("Enter number of clusters (k)", min_value=2, max_value=10, value=3)

# Take feature inputs dynamically
st.subheader("Enter Feature Values")
user_input = []
for feature in selected_features:
    value = st.number_input(f"{feature}", value=float(data[feature].median()))
    user_input.append(value)

# Scale user input
features = scaler.transform([user_input])

# ---------------------------
# Run Clustering
# ---------------------------
if st.button("Find Cluster"):

    if algorithm == "KMeans":
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(X_scaled)
        cluster = model.predict(features)[0]
        st.success(f"✅ This data point belongs to **Cluster {cluster}** (KMeans, k={k})")

    elif algorithm == "Hierarchical":
        model = AgglomerativeClustering(n_clusters=3)
        new_data = np.vstack([X_scaled, features])
        labels = model.fit_predict(new_data)
        cluster = labels[-1]
        st.success(f"✅ This data point belongs to **Cluster {cluster}** (Hierarchical)")

    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=0.5, min_samples=5)
        new_data = np.vstack([X_scaled, features])
        labels = model.fit_predict(new_data)
        cluster = labels[-1]
        if cluster == -1:
            st.error("🚨 This point is considered an OUTLIER (noise) by DBSCAN.")
        else:
            st.success(f"✅ This data point belongs to **Cluster {cluster}** (DBSCAN)")
