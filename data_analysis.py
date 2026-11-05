"""Compare customer clusters, visualize the results, and summarize segments."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from feature_engineering import SPENDING_COLUMNS, prepare_features
from model_comparison import compare_models


ROOT = Path(__file__).resolve().parent
CLEAN_DATA_PATH = ROOT / "datasets" / "cleaned_customers.csv"
OUTPUT_DIR = ROOT / "outputs"
PLOT_PATH = OUTPUT_DIR / "model_comparison.png"
SCORES_PATH = OUTPUT_DIR / "model_scores.csv"


def analyze_customers(customers=None):
    if customers is None:
        if not CLEAN_DATA_PATH.exists():
            raise FileNotFoundError(
                f"Cleaned data not found at {CLEAN_DATA_PATH}. "
                "Run data_cleaning.py first."
            )
        customers = pd.read_csv(CLEAN_DATA_PATH)

    pca_features, pca = prepare_features(customers)
    scores, labels_by_model = compare_models(pca_features)
    OUTPUT_DIR.mkdir(exist_ok=True)
    scores.to_csv(SCORES_PATH)

    figure, axes = plt.subplots(
        1, len(labels_by_model), figsize=(6 * len(labels_by_model), 5)
    )
    for axis, (name, labels) in zip(axes, labels_by_model.items()):
        scatter = axis.scatter(
            pca_features[:, 0],
            pca_features[:, 1],
            c=labels,
            cmap="viridis",
            alpha=0.7,
        )
        axis.set_title(name)
        axis.set_xlabel("Principal component 1")
        axis.set_ylabel("Principal component 2")
        axis.legend(*scatter.legend_elements(), title="Segment")
    figure.suptitle("Customer clusters compared in PCA space")
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=150)
    plt.close(figure)

    print(f"PCA retained {pca.explained_variance_ratio_.sum():.1%} of the variance.")
    print("\nClustering model comparison:")
    print(scores.round(3).to_string())

    customers = customers.copy()
    customers["Segment"] = labels_by_model["K-Means"]
    segment_averages = customers.groupby("Segment")[SPENDING_COLUMNS].mean().round(0)
    print("\nK-Means segments (highest average spending category):")
    for segment, averages in segment_averages.iterrows():
        top_category = averages.idxmax()
        customer_count = int((customers["Segment"] == segment).sum())
        print(
            f"  Segment {segment}: {customer_count} customers; "
            f"highest spend: {top_category}"
        )

    print(f"\nSaved comparison plot to {PLOT_PATH.relative_to(ROOT)}")
    print(f"Saved scores to {SCORES_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    analyze_customers()
