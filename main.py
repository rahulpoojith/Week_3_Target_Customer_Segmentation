"""Run the customer segmentation workflow."""

from data_analysis import analyze_customers
from data_cleaning import clean_data
from download_data import download_data


def main():
    download_data()
    customers = clean_data()
    analyze_customers(customers)


if __name__ == "__main__":
    main()
