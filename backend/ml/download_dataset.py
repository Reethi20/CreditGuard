
import kagglehub
import shutil
from pathlib import Path

# Download the dataset from Kaggle
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

print("Downloaded dataset location:", path)

# Define CreditGuard dataset directory
project_root = Path(__file__).resolve().parents[2]
raw_dir = project_root / "data" / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)

# Find the CSV file
csv_files = list(Path(path).glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV file found in downloaded dataset.")

# Copy dataset into CreditGuard
source_file = csv_files[0]
destination = raw_dir / "creditcard.csv"

shutil.copy2(source_file, destination)

print("Dataset successfully copied!")
print("Final location:", destination)
print("File size:", round(destination.stat().st_size / (1024 * 1024), 2), "MB")
