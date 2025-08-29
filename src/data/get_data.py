import os

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

raw_data_dir = "data/raw"
if not os.path.exists(raw_data_dir):
    os.makedirs(raw_data_dir)

dataset = "yashdevladdha/uber-ride-analytics-dashboard"
file_name = "ncr_ride_bookings.csv"

# Download directly to the target directory
api.dataset_download_file(dataset, file_name=file_name, path=raw_data_dir)

os.rename("data/raw/ncr_ride_bookings.csv", "data/raw/dataset.csv")

# # Rename to dataset.csv
# original_path = os.path.join(raw_data_dir, file_name)
# target_path = os.path.join(raw_data_dir, "dataset.csv")
# os.rename(original_path, target_path)
