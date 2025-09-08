import gradio as gr
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

# -----------------------------
# Load dataset
# -----------------------------
data = pd.read_csv("clustered_data.csv")

# Selected features
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

# Scaling
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Fit DBSCAN and Hierarchical once
hierarchical_model = AgglomerativeClustering(n_clusters=4).fit(X_scaled)
dbscan_model = DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)
dbscan_core_samples = dbscan_model.components_
dbscan_core_labels = dbscan_model.labels_[dbscan_model.core_sample_indices_]

# -----------------------------
# Prediction function
# -----------------------------
def predict_cluster(username, password, algorithm, k, *features):
    if username != "admin" or password != "1234":
        return "❌ Invalid login. Please try again.", None, None

    try:
        features = list(map(float, features))
        features_scaled = scaler.transform([features])

        if algorithm == "KMeans":
            model = KMeans(n_clusters=int(k), random_state=42).fit(X_scaled)
            cluster = model.predict(features_scaled)[0]
            labels = model.labels_

            sil = silhouette_score(X_scaled, labels)
            db = davies_bouldin_score(X_scaled, labels)
            ch = calinski_harabasz_score(X_scaled, labels)
            inertia = model.inertia_

            metrics_df = pd.DataFrame([{
                "Model": "KMeans",
                "Silhouette": sil,
                "Davies-Bouldin": db,
                "Calinski-Harabasz": ch,
                "Inertia": inertia,
                "Clusters": int(k)
            }])

            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            plt.figure(figsize=(6, 5))
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=20)
            plt.scatter(*pca.transform(features_scaled).T, c='red', s=80, marker='X', label='New Point')
            plt.title(f"KMeans Clustering (Predicted Cluster: {cluster})")
            plt.legend()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            plt.close()

            return (
                f"✅ This data point belongs to **Cluster {cluster}** (KMeans, k={k})",
                metrics_df,
                img
            )

        elif algorithm == "Hierarchical":
            new_data = np.vstack([X_scaled, features_scaled])
            labels = AgglomerativeClustering(n_clusters=4).fit_predict(new_data)
            cluster = labels[-1]

            sil = silhouette_score(new_data[:-1], labels[:-1])
            db = davies_bouldin_score(new_data[:-1], labels[:-1])
            ch = calinski_harabasz_score(new_data[:-1], labels[:-1])

            metrics_df = pd.DataFrame([{
                "Model": "Hierarchical",
                "Silhouette": sil,
                "Davies-Bouldin": db,
                "Calinski-Harabasz": ch,
                "Inertia": None,
                "Clusters": 4
            }])

            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(new_data)
            plt.figure(figsize=(6, 5))
            plt.scatter(X_pca[:-1, 0], X_pca[:-1, 1], c=labels[:-1], cmap='tab10', s=20)
            plt.scatter(*X_pca[-1], c='red', s=80, marker='X', label='New Point')
            plt.title(f"Hierarchical Clustering (Predicted Cluster: {cluster})")
            plt.legend()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            plt.close()

            return (
                f"✅ This data point belongs to **Cluster {cluster}** (Hierarchical)",
                metrics_df,
                img
            )

        elif algorithm == "DBSCAN":
            dists = euclidean_distances(features_scaled, dbscan_core_samples)
            nearest_idx = np.argmin(dists)
            nearest_dist = dists[0, nearest_idx]
            cluster = dbscan_core_labels[nearest_idx]

            # Create a new dataset for plotting
            X_combined = np.vstack([X_scaled, features_scaled])
            labels = np.append(dbscan_model.labels_, cluster if nearest_dist <= dbscan_model.eps else -1)

            # Metrics (only if point is NOT outlier)
            if nearest_dist <= dbscan_model.eps and cluster != -1:
                sil = silhouette_score(X_scaled, dbscan_model.labels_)
                db = davies_bouldin_score(X_scaled, dbscan_model.labels_)
                ch = calinski_harabasz_score(X_scaled, dbscan_model.labels_)

                metrics_df = pd.DataFrame([{
                    "Model": "DBSCAN",
                    "Silhouette": sil,
                    "Davies-Bouldin": db,
                    "Calinski-Harabasz": ch,
                    "Inertia": None,
                    "Clusters": len(set(dbscan_model.labels_)) - (1 if -1 in dbscan_model.labels_ else 0)
                }])
            else:
                metrics_df = None

            # PCA plot
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_combined)
            plt.figure(figsize=(6, 5))
            plt.scatter(X_pca[:-1, 0], X_pca[:-1, 1], c=dbscan_model.labels_, cmap='tab10', s=20)
            plt.scatter(*X_pca[-1], c='red', s=80, marker='X', label='New Point')
            title = f"DBSCAN Clustering"
            if nearest_dist > dbscan_model.eps:
                title += " (Outlier)"
            else:
                title += f" (Predicted Cluster: {cluster})"
            plt.title(title)
            plt.legend()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            plt.close()

            if nearest_dist <= dbscan_model.eps:
                return (
                    f"✅ This data point belongs to **Cluster {cluster}** (DBSCAN, dist={nearest_dist:.2f})",
                    metrics_df,
                    img
                )
            else:
                return (
                    f"🚨 This point is considered an **OUTLIER** by DBSCAN (dist={nearest_dist:.2f})",
                    None,
                    img
                )

    except Exception as e:
        return f"⚠️ Error: {str(e)}", None, None

# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks() as demo:
    with gr.Tab("🔑 Login"):
        gr.Markdown("## Login to Access Clustering App")
        username = gr.Textbox(label="Username")
        password = gr.Textbox(label="Password", type="password")
        algorithm = gr.Dropdown(["KMeans", "Hierarchical", "DBSCAN"], label="Select Algorithm")
        k_value = gr.Number(label="Number of Clusters (for KMeans)", value=3)

        inputs = []
        with gr.Accordion("Enter Feature Values", open=False):
            for feature in selected_features:
                inputs.append(gr.Number(label=feature, value=float(data[feature].median())))

        btn = gr.Button("Find Cluster")
        result_text = gr.Textbox(label="Cluster Result")
        metrics_table = gr.Dataframe(label="Clustering Metrics")
        cluster_plot = gr.Image(label="Cluster Visualization")

        btn.click(
            fn=predict_cluster,
            inputs=[username, password, algorithm, k_value] + inputs,
            outputs=[result_text, metrics_table, cluster_plot],
        )

# -----------------------------
# Run the app
# -----------------------------
if __name__ == "__main__":
    demo.launch(debug=True)