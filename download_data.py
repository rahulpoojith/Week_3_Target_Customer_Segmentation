"""Download the UCI Wholesale Customers dataset into datasets/."""

from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent
DATA_URL = "https://archive.ics.uci.edu/static/public/292/wholesale+customers.zip"
DATA_PATH = ROOT / "datasets" / "wholesale_customers.csv"


def download_data():
    DATA_PATH.parent.mkdir(exist_ok=True)
    with urlopen(DATA_URL, timeout=30) as response:
        archive = ZipFile(BytesIO(response.read()))

    csv_name = next(name for name in archive.namelist() if name.endswith(".csv"))
    DATA_PATH.write_bytes(archive.read(csv_name))
    print(f"Downloaded dataset to {DATA_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    download_data()
