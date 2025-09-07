import gradio as gr
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import RobustScaler
from sklearn.metrics.pairwise import euclidean_distances

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
# Train models
# ---------------------------
hierarchical_model = AgglomerativeClustering(n_clusters=4).fit(X_scaled)
dbscan_model = DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)

# Save DBSCAN core samples for distance checking
dbscan_core_samples = dbscan_model.components_
dbscan_core_labels = dbscan_model.labels_[dbscan_model.core_sample_indices_]

# ---------------------------
# Prediction Function
# ---------------------------
def predict_cluster(username, password, algorithm, k, *features):
    # ✅ Login credentials
    if username != "admin" or password != "1234":
        return "❌ Invalid login. Please try again."

    # Scale input
    features_scaled = scaler.transform([features])

    if algorithm == "KMeans":
        model = KMeans(n_clusters=int(k), random_state=42).fit(X_scaled)
        cluster = model.predict(features_scaled)[0]
        return f"✅ This data point belongs to **Cluster {cluster}** (KMeans, k={k})"

    elif algorithm == "Hierarchical":
        new_data = np.vstack([X_scaled, features_scaled])
        labels = AgglomerativeClustering(n_clusters=4).fit_predict(new_data)
        cluster = labels[-1]
        return f"✅ This data point belongs to **Cluster {cluster}** (Hierarchical)"

    elif algorithm == "DBSCAN":
        # Find nearest DBSCAN core point
        dists = euclidean_distances(features_scaled, dbscan_core_samples)
        nearest_idx = np.argmin(dists)
        nearest_dist = dists[0, nearest_idx]
        cluster = dbscan_core_labels[nearest_idx]

        if nearest_dist <= dbscan_model.eps:
            return f"✅ This data point belongs to **Cluster {cluster}** (DBSCAN, dist={nearest_dist:.2f})"
        else:
            return f"🚨 This point is considered an OUTLIER (noise) by DBSCAN."

# ---------------------------
# Gradio UI
# ---------------------------
with gr.Blocks() as demo:
    with gr.Tab("🔑 Login"):
        gr.Markdown("## Login to Access Clustering App")
        username = gr.Textbox(label="Username")
        password = gr.Textbox(label="Password", type="password")
        algorithm = gr.Dropdown(["KMeans", "Hierarchical", "DBSCAN"], label="Select Algorithm")

        # Input for k (only used if KMeans is selected)
        k_value = gr.Number(label="Number of Clusters (k for KMeans)", value=3)

        inputs = []
        with gr.Accordion("Enter Feature Values", open=False):
            for feature in selected_features:
                inputs.append(gr.Number(label=feature, value=float(data[feature].median())))

        btn = gr.Button("Find Cluster")
        output = gr.Textbox(label="Result")

        btn.click(
            fn=predict_cluster,
            inputs=[username, password, algorithm, k_value] + inputs,
            outputs=output,
        )

# ---------------------------
# Launch
# ---------------------------
if __name__ == "__main__":
    demo.launch()
