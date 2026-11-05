# Week 3: Customer Segmentation with PCA

This project explores customer groups using **unsupervised learning**. It uses
the UCI Wholesale Customers dataset, applies PCA to customer spending data, and
compares two simple clustering methods: K-Means and Agglomerative Clustering.
It does not train a neural network or use the known customer channel to form
the clusters.

## What the project does

1. **Download data:** Fetches the Wholesale Customers CSV from the UCI Machine
   Learning Repository and saves it in `datasets/`.
2. **Clean data:** Keeps the channel, region, and six annual spending columns;
   removes duplicate, incomplete, or invalid rows; saves the cleaned CSV.
3. **Prepare features:** Standardizes the six spending columns so large-value
   categories do not dominate, then applies PCA. PCA keeps enough components to
   represent 95% of the variance.
4. **Compare clusters:** Tests K-Means and Agglomerative Clustering with 2 to 6
   groups using the PCA features, then keeps the cluster count with the best
   silhouette score for each method.
5. **Review results:** Prints each model's silhouette score and summarizes
   K-Means segments by customer count and highest average spending category.

## Set up

From the project directory, create and activate a virtual environment, then
install the required packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

## Run the project

Run the complete workflow with:

```bash
python main.py
```

The dataset is downloaded automatically the first time. The script prints the
number of cleaned customers, PCA variance retained, model scores, and a short
interpretation of the K-Means segments.

You can also run the steps separately, in this order:

```bash
python download_data.py
python data_cleaning.py
python data_analysis.py
```

## Understanding the comparison

Both clustering methods try the same small range of cluster counts (2 through
6), and each method's highest-scoring result is shown. The **silhouette score**
ranges from -1 to 1: a higher score generally indicates that customers are
closer to their own segment than to other segments. It is an internal
clustering measure and does not require known labels.

The plot shows the first two principal components so the groups can be viewed
on a 2D chart. Clustering uses all PCA components needed to retain 95% of the
variance, not only the two displayed in the chart. Cluster numbers are
arbitrary labels, not rankings.

## Project files

- `main.py` runs the full workflow in order.
- `download_data.py` downloads the UCI dataset into `datasets/`.
- `data_cleaning.py` checks the required columns and saves cleaned customer
  records.
- `feature_engineering.py` standardizes spending columns and applies PCA.
- `model_comparison.py` fits K-Means and Agglomerative Clustering and compares
  their silhouette scores.
- `data_analysis.py` creates the plot and prints a simple K-Means segment
  summary.
- `requirements.txt` lists the Python packages used by the project.
- `datasets/` stores the downloaded and cleaned CSV files.
- `outputs/` stores the generated plot and model score CSV.

## Generated results

After running `main.py`, look in `outputs/` for:

- `model_comparison.png` — side-by-side PCA visualizations of the two models.
- `model_scores.csv` — cluster counts and silhouette scores.

The cleaned dataset is saved as `datasets/cleaned_customers.csv`. The raw
dataset can be downloaded again by running `python download_data.py`.

## Dataset

The [UCI Wholesale Customers dataset](https://archive.ics.uci.edu/dataset/292/wholesale+customers)
contains annual spending for wholesale customers across Fresh, Milk, Grocery,
Frozen, Detergents/Paper, and Delicassen categories. Channel and Region are
included in the source data; this project uses the spending columns to form
clusters.

## Results

The cleaned dataset contains **440 customers**. PCA retains approximately
**99.0%** of the spending-data variance in the components used for clustering.
The silhouette scores from this run were:

The model comparison tries between 2 and 6 clusters and reports the best
silhouette score found for each method. In the current run:

| Model | Selected clusters | Silhouette score |
|---|---:|---:|
| K-Means | 2 | 0.556 |
| Agglomerative Clustering | 2 | 0.539 |

K-Means groups 53 customers in the segment with the highest average Grocery
spending and 387 customers in the segment with the highest average Fresh
spending. The plot and CSV contain the latest run's results. Results depend on
the dataset and configured random seed. Cluster labels are arbitrary; their
numbers do not indicate an order or rank.

## Conclusion

PCA makes it easier to explore customer spending patterns with fewer
dimensions, while the small silhouette-score search provides a straightforward
way to choose a cluster count for each method. In this run, K-Means had a
slightly higher silhouette score than Agglomerative Clustering. The score is a
useful starting point, but it does not guarantee that the groups are actionable
or balanced. Review segment sizes and spending averages before using them for
business decisions; these clusters are exploratory, not definitive customer
types.
