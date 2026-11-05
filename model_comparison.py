"""Compare K-Means and Agglomerative clustering."""

import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score


def compare_models(features):
    models = {
        "K-Means": KMeans(n_clusters=3, random_state=42, n_init=10),
        "Agglomerative": AgglomerativeClustering(n_clusters=3),
    }

    labels_by_model = {}
    scores = []
    for name, model in models.items():
        labels = model.fit_predict(features)
        labels_by_model[name] = labels
        scores.append(
            {
                "Model": name,
                "Clusters": len(set(labels)),
                "Silhouette (higher is better)": silhouette_score(features, labels),
            }
        )

    return pd.DataFrame(scores).set_index("Model"), labels_by_model
