import json
import logging
from torch.utils.data import Dataset
import datetime

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

class ETFDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        sample = self.data[index]
        # Label
        label = sample.get("ETFTicker")
        if label is None:
            logging.warning(f"Missing 'ETFTicker' in sample at index {index}")
            label = "Unknown"  # Default value or handle accordingly

        # Numeric features
        tot_ret_ytd = sample["Tot Ret Ytd"]
        tot_ret_1y = sample["Tot Ret 1Y"]

        # Categorical features
        fund_asset_class_focus = sample["Fund Asset Class Focus"]

        # Summary features
        summary = sample.get("Summary")
        if summary is None:
            logging.warning(f"Missing 'Summary' in sample at index {index}")
            summary = {}  # Default to an empty dictionary
        class_assets = summary.get("Class Assets (MLN USD)", 0)
        fund_assets = summary.get("Fund Assets (MLN USD)", 0)
        expense_ratio = summary.get("Expense Ratio", 0)
        ytd_return = summary.get("YTD Return", 0)
        twelve_month_yield = summary.get("12M Yld", 0)
        thirty_day_vol = summary.get("30D Vol", 0)
        ytd_flow = summary.get("YTD Flow", 0)
        one_month_flow = summary.get("1M Flow", 0)
        one_year_nav_trk_error = summary.get("1 Yr NAV Trk Error", 0)
        holdings = summary.get("Holdings", 0)
        primary = summary.get("Primary", 0)
        cross = summary.get("Cross", 0)

        # Convert categorical features to numeric representations if needed
        # For example, you can use one-hot encoding or label encoding

        # Create a dictionary of features
        features = {
            "tot_ret_ytd": tot_ret_ytd,
            "tot_ret_1y": tot_ret_1y,
            "fund_asset_class_focus": fund_asset_class_focus,
            "class_assets": class_assets,
            "fund_assets": fund_assets,
            "expense_ratio": expense_ratio,
            "ytd_return": ytd_return,
            "twelve_month_yield": twelve_month_yield,
            "thirty_day_vol": thirty_day_vol,
            "ytd_flow": ytd_flow,
            "one_month_flow": one_month_flow,
            "one_year_nav_trk_error": one_year_nav_trk_error,
            "holdings": holdings,
            "primary": primary,
            "cross": cross
        }

        return {
            "etf_ticker": label,
            "features": features
        }

    def __len__(self):
        return len(self.data)

    def timestamp_to_seconds(self, timestamp):
        # Convert timestamp string to datetime object
        dt = datetime.datetime.strptime(timestamp, "%Y:%m:%d, %H:%M:%S")

        # Convert datetime object to seconds since a reference point
        seconds = int((dt - datetime.datetime(1970, 1, 1)).total_seconds())

        return seconds

    def timestamp_to_seconds(self, timestamp):
        # Convert timestamp string to datetime object
        dt = datetime.datetime.strptime(timestamp, "%Y:%m:%d, %H:%M:%S")

        # Convert datetime object to seconds since a reference point
        seconds = int((dt - datetime.datetime(1970, 1, 1)).total_seconds())

        return seconds
    def process_time_series(self, time_series_data):
        # Preprocess and convert time series data to a sequence of timestamp-value pairs
        processed_sequence = ...
        return processed_sequence

def main(json_file):
    # Load the JSON data from the file
    with open(json_file, 'r') as file:
        json_data = json.load(file)

    # Create an instance of the ETFDataset
    dataset = ETFDataset(json_data)

    # Print the first 10 samples
    for i in range(10):
        sample = dataset[i]
        print(f"Sample {i+1}:")
        print(f"  ETF Ticker: {sample['etf_ticker']}")
        print(f"  Features:")
        for feature, value in sample['features'].items():
            print(f"    {feature}: {value}")
        print()

if __name__ == '__main__':
    json_file = '/home/oleg/Documents/courses/Stanford/CS224N/FinalProject/code/FolioLLM/data/etf_data.json'  # Replace with the path to your JSON file
    main(json_file)