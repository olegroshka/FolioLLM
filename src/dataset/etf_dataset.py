import json
from torch.utils.data import Dataset

import datetime

class ETFDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        sample = self.data[index]
        label = sample["ETFTicker"]  # Label

        # Numeric features
        tot_ret_ytd = sample["Tot Ret Ytd"]
        tot_ret_1y = sample["Tot Ret 1Y"]

        # Categorical features
        fund_asset_class_focus = sample["Fund Asset Class Focus"]

        # Summary features
        summary = sample["Summary"]
        class_assets = summary["Class Assets (MLN USD)"]
        fund_assets = summary["Fund Assets (MLN USD)"]
        expense_ratio = summary["Expense Ratio"]
        ytd_return = summary["YTD Return"]
        twelve_month_yield = summary["12M Yld"]
        thirty_day_vol = summary["30D Vol"]
        ytd_flow = summary["YTD Flow"]
        one_month_flow = summary["1M Flow"]
        one_year_nav_trk_error = summary["1 Yr NAV Trk Error"]
        holdings = summary["Holdings"]
        primary = summary["Primary"]
        cross = summary["Cross"]

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