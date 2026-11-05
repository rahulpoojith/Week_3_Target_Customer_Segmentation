"""Clean the downloaded customer data and save the cleaned CSV."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = ROOT / "datasets" / "wholesale_customers.csv"
CLEAN_DATA_PATH = ROOT / "datasets" / "cleaned_customers.csv"
SPENDING_COLUMNS = [
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen",
]
REQUIRED_COLUMNS = ["Channel", "Region", *SPENDING_COLUMNS]


def clean_data():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {RAW_DATA_PATH}. Run download_data.py first."
        )

    customers = pd.read_csv(RAW_DATA_PATH)
    missing_columns = set(REQUIRED_COLUMNS) - set(customers.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

    customers = customers[REQUIRED_COLUMNS].copy()
    customers = customers.drop_duplicates().dropna(subset=REQUIRED_COLUMNS)
    customers[REQUIRED_COLUMNS] = customers[REQUIRED_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    customers = customers.dropna(subset=REQUIRED_COLUMNS)
    customers = customers[
        customers[SPENDING_COLUMNS].ge(0).all(axis=1)
        & customers["Channel"].isin([1, 2])
    ]

    CLEAN_DATA_PATH.parent.mkdir(exist_ok=True)
    customers.to_csv(CLEAN_DATA_PATH, index=False)
    print(
        f"Cleaned data: {len(customers)} customers saved to "
        f"{CLEAN_DATA_PATH.relative_to(ROOT)}"
    )
    return customers


if __name__ == "__main__":
    clean_data()
