"""Reference solution: unsupervised customer segmentation with PCA + KMeans,
cross-checked against Hierarchical clustering and DBSCAN.

Run:
    python data/generate_data.py
    python analysis.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).parent / "data" / "customers.csv"
CHARTS_DIR = Path(__file__).parent / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

FEATURES = ["recency_days", "frequency", "monetary", "avg_basket_size"]


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].to_numpy()
    X_scaled = StandardScaler().fit_transform(X)

    # --- elbow + silhouette to choose k ---
    inertias, sil_scores = [], []
    ks = range(2, 9)
    for k in ks:
        model = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X_scaled)
        inertias.append(model.inertia_)
        sil_scores.append(silhouette_score(X_scaled, model.labels_))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(ks, inertias, marker="o")
    axes[0].set_title("Elbow (inertia)")
    axes[1].plot(ks, sil_scores, marker="o")
    axes[1].set_title("Silhouette score")
    fig.savefig(CHARTS_DIR / "k_selection.png")
    plt.close(fig)

    best_k = list(ks)[int(np.argmax(sil_scores))]
    print("k chosen by silhouette:", best_k)

    # --- final KMeans + PCA visualization ---
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit(X_scaled)
    df["segment"] = kmeans.labels_

    X_pca = PCA(n_components=2).fit_transform(X_scaled)
    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap="tab10", s=15)
    plt.title(f"KMeans segments (k={best_k}) in PCA space")
    plt.savefig(CHARTS_DIR / "pca_segments.png")
    plt.close()

    # --- second opinions ---
    agg_labels = AgglomerativeClustering(n_clusters=best_k).fit_predict(X_scaled)
    dbscan_labels = DBSCAN(eps=0.6, min_samples=10).fit_predict(X_scaled)
    print("Agglomerative cluster sizes:", np.bincount(agg_labels))
    print("DBSCAN cluster sizes (label -1 = noise):", np.unique(dbscan_labels, return_counts=True))

    # --- segment profiles ---
    profile = df.groupby("segment")[FEATURES].mean().round(1)
    profile["count"] = df.groupby("segment").size()
    print("\n=== Segment profiles ===")
    print(profile)


if __name__ == "__main__":
    main()
