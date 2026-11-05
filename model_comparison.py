"""Compare K-Means and Agglomerative clustering."""

import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score


def compare_models(features):
    model_builders = {
        "K-Means": lambda count: KMeans(
            n_clusters=count, random_state=42, n_init=10
        ),
        "Agglomerative": lambda count: AgglomerativeClustering(n_clusters=count),
    }

    labels_by_model = {}
    scores = []
    for name, build_model in model_builders.items():
        best_score = -1
        best_count = None
        best_labels = None

        for count in range(2, 7):
            labels = build_model(count).fit_predict(features)
            score = silhouette_score(features, labels)
            if score > best_score:
                best_score = score
                best_count = count
                best_labels = labels

        labels_by_model[name] = best_labels
        scores.append(
            {
                "Model": name,
                "Clusters": best_count,
                "Silhouette (higher is better)": best_score,
            }
        )

    return pd.DataFrame(scores).set_index("Model"), labels_by_model
