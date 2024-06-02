import json
import logging
from torch.utils.data import Dataset

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

etf_text_template = """ETF Details:
Ticker: {ticker}
Bloomberg Ticker: {bbg_ticker}
Name: {etf_name}
Description: {description}
Type: {fund_type}
Manager: {manager}

{etf_name} ({ticker}) Fund Classification:
    Asset Class Focus: {asset_class_focus}
    Asset Group: {fund_asset_group}
    Industry Focus: {fund_industry_focus}
    Geographical Focus: {fund_geographical_focus}
    Objective: {fund_objective}
    Economic Association: {economic_association}
    Strategy: {fund_strategy}
    Market Cap Focus: {fund_market_cap_focus}
    Style: {fund_style}

{etf_name} ({ticker}) Summary:
    Class Assets (MLN USD): {class_assets}
    Fund Assets (MLN USD): {fund_assets}
    Expense Ratio: {expense_ratio}
    Year-To-Date Return: {year_to_date_return}
    30 Days Volatility: {volume_30d}
    Year-To-Date Flow: {ytd_flow}
    1 Month Flow: {flow_1m}
    1 Year NAV Tracking Error: {nav_trk_error}
    Holdings: {holdings}
    Primary: {primary}
    Cross: {primary}

{etf_name} ({ticker}) Performance:
    1 Day Return: {return_1d}
    Month-To-Date Return: {return_mtd}
    Year-To-Date Return: {ytd_return}
    3 Years Return: {return_3y}

{etf_name} ({ticker}) Liquidity:
    1 Day Volume: {volume_1d}
    Aggregated Volume: {aggregated_volume}
    Aggregated Value Traded: {total_value_traded}
    Bid Ask Spread: {bid_ask_spread}

{etf_name} ({ticker}) Expense:
    Expense Ratio: {expense_ratio}
    Fund Manager Stated Fee: {expense_ratio}
    Average Bid Ask Spread: {avg_bid_ask_spread}
    1 Year NAV Tracking Error: {nav_trk_error}
    Premium: {premium}
    52 Weeks Average Premium: {premium_52w}

{etf_name} ({ticker}) Flow:
    Currency: {currency}
    responseOAS Effective Duration: {oas_effective_duration}
    OAS Duration Coverage Ratio: {oas_duration_coverage_ratio}
    Options Available: {options_available}
    Payment Type: {payment_type}

{etf_name} ({ticker}) Regulatory:
    Fund Type: {fund_type}
    Structure: {structure}
    Index Weight: {index_weight}
    Use Derivative: {use_derivative}
    Tax Form: {tax_form}
    UCITS: {ucits}
    UK Reporting: {uk_reporting}
    SFC: {sfc}
    China: {china}
    Leverage: {leverage}
    Inception Date: {inception_date}

{etf_name} ({ticker}) Industry Exposure:
    Materials: {materials}
    Communications: {communications}
    Consumer Cyclical: {consumer_cyclical}
    Consumer Non-Cyclical: {consumer_non_cyclical}
    Energy: {energy}
    Financials: {financials}
    Industrials: {industrials}
    Technology: {technology}
    Utilities: {utilities}

{etf_name} ({ticker}) Geographical Exposure:
    North America: {north_america}
    Western Europe: {western_europe}
    Asia Pacific: {asia_pacific}
"""

class ETFTextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        sample = self.data[index]
        text = self.create_etf_text(sample)
        return {
            "text": text
        }

    def __len__(self):
        return len(self.data)

    def create_etf_text(self, sample):
        formatted_sample = {key: custom_format(value, key) for key, value in sample.items()}
        return etf_text_template.format(**formatted_sample)


def custom_format(value, key):
    if value == "N":
        return "No"
    elif value == "Y":
        return "Yes"
    return value

def main(json_file):
    # Load the JSON data from the file
    with open(json_file, 'r') as file:
        json_data = json.load(file)

    # Create an instance of the ETFTextDataset
    dataset = ETFTextDataset(json_data)

    # Print the first 10 samples
    for i in range(10):
        sample = dataset[i]
        print(f"Sample {i+1}:")
        print(sample['text'])
        print()

if __name__ == '__main__':
    json_file = '../../data/etf_data_v3_plain.json'
    main(json_file)

