# Customer Segmentation

A small unsupervised-learning project using the UCI Wholesale Customers dataset.
It cleans customer spending data, scales the features, reduces them with PCA,
then compares two simple clustering models: K-Means and Agglomerative
Clustering.

## Set up

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate it with `.venv\Scripts\activate`.

## Run

```bash
python main.py
```

The script downloads the dataset from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/292/wholesale+customers),
cleans it, and runs PCA and clustering.

## Project files

- `download_data.py` downloads the dataset into `datasets/`.
- `data_cleaning.py` removes duplicate, incomplete, or invalid customer rows.
- `feature_engineering.py` scales spending columns and applies PCA.
- `model_comparison.py` compares K-Means and Agglomerative clustering using the
  silhouette score (higher is better).
- `data_analysis.py` visualizes the clusters and summarizes K-Means segments.
- `main.py` runs the workflow in order.

The cleaned data is saved in `datasets/`. A PCA plot comparing both models and
the silhouette scores are saved in `outputs/`. The score measures how well
separated the clusters are and does not use customer labels.

# Week3_Target_Customer_Segmentation_PCA
