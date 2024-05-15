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

        etf_name = self.get_categorical_feature(sample, "Name")
        manager = self.get_categorical_feature(sample, "Manager")

        # Numeric features
        tot_ret_ytd = self.get_numeric_feature(sample, "Tot Ret Ytd")
        tot_ret_1y = self.get_numeric_feature(sample, "Tot Ret 1Y")

        # Categorical features
        fund_asset_class_focus = self.get_categorical_feature(sample, "Fund Asset Class Focus")

        # Summary features
        summary = sample.get("Summary")
        if summary is None:
            logging.warning(f"Missing 'Summary' in sample at index {index}")
            summary = {}  # Default to an empty dictionary
        class_assets = self.get_numeric_feature(summary, "Class Assets (MLN USD)")
        fund_assets = self.get_numeric_feature(summary, "Fund Assets (MLN USD)")
        expense_ratio = self.get_numeric_feature(summary, "Expense Ratio")
        ytd_return = self.get_numeric_feature(summary, "YTD Return")
        twelve_month_yield = self.get_numeric_feature(summary, "12M Yld")
        thirty_day_vol = self.get_numeric_feature(summary, "30D Vol")
        ytd_flow = self.get_numeric_feature(summary, "YTD Flow")
        one_month_flow = self.get_numeric_feature(summary, "1M Flow")
        one_year_nav_trk_error = self.get_numeric_feature(summary, "1 Yr NAV Trk Error")
        holdings = self.get_numeric_feature(summary, "Holdings")
        primary = self.get_categorical_feature(summary, "Primary")
        cross = self.get_categorical_feature(summary, "Cross")

        # Convert categorical features to numeric representations if needed
        # For example, you can use one-hot encoding or label encoding

        # Create a dictionary of features
        features = {
            "etf_name": etf_name,
            "manager": manager,
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

    def get_numeric_feature(self, data, feature_name):
        value = data.get(feature_name)
        if value is None or value == "--":
            return 0.0  # Default value for missing numeric features
        if isinstance(value, str):
            try:
                return float(
                    value.replace("%", "").replace(",", ""))  # Remove '%' and ',' characters and convert to float
            except ValueError:
                return 0.0  # Default value if conversion fails
        elif isinstance(value, (int, float)):
            return float(value)  # Return the value as a float
        else:
            return 0.0  # Default value for unsupported types

    def get_categorical_feature(self, data, feature_name):
        value = data.get(feature_name)
        if value is None:
            return ""  # Default value for missing categorical features
        return str(value)  # Convert to string

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