"""Scale spending data and reduce it with PCA."""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


SPENDING_COLUMNS = [
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen",
]


def prepare_features(customers):
    """Scale spending columns and retain 95% of their variance with PCA."""
    scaled = StandardScaler().fit_transform(customers[SPENDING_COLUMNS])
    pca = PCA(n_components=0.95)
    pca_features = pca.fit_transform(scaled)
    return pca_features, pca
