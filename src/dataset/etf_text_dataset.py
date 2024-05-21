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
    {ticker} Asset Class Focus: {asset_class_focus}
    {ticker} Asset Group: {fund_asset_group}
    {ticker} Industry Focus: {fund_industry_focus}
    {ticker} Geographical Focus: {fund_geographical_focus}
    {ticker} Objective: {fund_objective}
    {ticker} Economic Association: {economic_association}
    {ticker} Strategy: {fund_strategy}
    {ticker} Market Cap Focus: {fund_market_cap_focus}
    {ticker} Style: {fund_style}

{etf_name} ({ticker}) Summary:
    {ticker} Class Assets (MLN USD): {class_assets}
    {ticker} Fund Assets (MLN USD): {fund_assets}
    {ticker} Expense Ratio: {expense_ratio}
    {ticker} Year-To-Date Return: {year_to_date_return}
    {ticker} 30 Days Volatility: {volume_30d}
    {ticker} Year-To-Date Flow: {ytd_flow}
    {ticker} 1 Month Flow: {flow_1m}
    {ticker} 1 Year NAV Tracking Error: {nav_trk_error}
    {ticker} Holdings: {holdings}
    {ticker} Primary: {primary}
    {ticker} Cross: {primary}

{etf_name} ({ticker}) Performance:
    {ticker} 1 Day Return: {return_1d}
    {ticker} Month-To-Date Return: {return_mtd}
    {ticker} Year-To-Date Return: {ytd_return}
    {ticker} 3 Years Return: {return_3y}

{etf_name} ({ticker}) Liquidity:
    {ticker} 1 Day Volume: {volume_1d}
    {ticker} Aggregated Volume: {aggregated_volume}
    {ticker} Aggregated Value Traded: {total_value_traded}
    {ticker} Bid Ask Spread: {bid_ask_spread}

{etf_name} ({ticker}) Expense:
    {ticker} Expense Ratio: {expense_ratio}
    {ticker} Fund Manager Stated Fee: {expense_ratio}
    {ticker} Average Bid Ask Spread: {avg_bid_ask_spread}
    {ticker} 1 Year NAV Tracking Error: {nav_trk_error}
    {ticker} Premium: {premium}
    {ticker} 52 Weeks Average Premium: {premium_52w}

{etf_name} ({ticker}) Flow:
    {ticker} Currency: {currency}
    {ticker} OAS Effective Duration: {oas_effective_duration}
    {ticker} OAS Duration Coverage Ratio: {oas_duration_coverage_ratio}
    {ticker} Options Available: {options_available}
    {ticker} Payment Type: {payment_type}

{etf_name} ({ticker}) Regulatory:
    {ticker} Fund Type: {fund_type}
    {ticker} Structure: {structure}
    {ticker} Index Weight: {index_weight}
    {ticker} Use Derivative: {use_derivative}
    {ticker} Tax Form: {tax_form}
    {ticker} UCITS: {ucits}
    {ticker} UK Reporting: {uk_reporting}
    {ticker} SFC: {sfc}
    {ticker} China: {china}
    {ticker} Leverage: {leverage}
    {ticker} Inception Date: {inception_date}

{etf_name} ({ticker}) Industry Exposure:
    {ticker} Materials: {materials}
    {ticker} Communications: {communications}
    {ticker} Consumer Cyclical: {consumer_cyclical}
    {ticker} Consumer Non-Cyclical: {consumer_non_cyclical}
    {ticker} Energy: {energy}
    {ticker} Financials: {financials}
    {ticker} Industrials: {industrials}
    {ticker} Technology: {technology}
    {ticker} Utilities: {utilities}

{etf_name} ({ticker}) Geographical Exposure:
    {ticker} North America: {north_america}
    {ticker} Western Europe: {western_europe}
    {ticker} Asia Pacific: {asia_pacific}
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

