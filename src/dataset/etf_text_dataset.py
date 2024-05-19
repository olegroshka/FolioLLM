import json
import logging
from torch.utils.data import Dataset

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

class ETFTextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        sample = self.data[index]
        text_snippet = self.create_text_snippet(sample)
        return {
            "text": text_snippet
        }

    def __len__(self):
        return len(self.data)

    def create_text_snippet(self, sample):
        snippet = f"""ETF Details:
Ticker: {sample.get("Ticker", "N/A")}
Bloomberg Ticker: {sample.get("BBG Ticker", "N/A")}
FIGI: {sample.get("FIGI", "N/A")}
Name: {sample.get("Name", "N/A")}
Description: {sample.get("Description", "N/A")}
Type: {sample.get("Type", "N/A")}
Domicile: {sample.get("Domicile", "N/A")}
Manager: {sample.get("Manager", "N/A")}

 {sample.get("Ticker", "N/A")} Fund Classification:
    Asset Class Focus: {sample.get("Fund Asset Class Focus", "N/A")}
    Asset Group: {sample.get("Fund Asset Group", "N/A")}
    Industry Focus: {sample.get("Fund Industry Focus", "N/A")}
    Geographical Focus: {sample.get("Fund Geographical Focus", "N/A")}
    Objective: {sample.get("Fund Objective", "N/A")}
    Economic Association: {sample.get("Economic Association", "N/A")}
    Strategy: {sample.get("Fund Strategy", "N/A")}
    Market Cap Focus: {sample.get("Fund Market Cap Focus", "N/A")}
    Style: {sample.get("Fund Style", "N/A")}

{sample.get("Ticker", "N/A")} Summary:
    Class Assets (MLN USD): {self.get_value(sample, "Summary", "Class Assets (MLN USD)")}
    Fund Assets (MLN USD): {self.get_value(sample, "Summary", "Fund Assets (MLN USD)")}
    Expense Ratio: {self.get_value(sample, "Summary", "Expense Ratio")}
    Year-To-Date Return: {self.get_value(sample, "Summary", "Year-To-Date Return")}
    12 Months Yield: {self.get_value(sample, "Summary", "12Months Yield")}
    30 Days Volatility: {self.get_value(sample, "Summary", "30Days Volatility")}
    Year-To-Date Flow: {self.get_value(sample, "Summary", "Year-To-Date Flow")}
    1 Month Flow: {self.get_value(sample, "Summary", "1Month Flow")}
    1 Year NAV Tracking Error: {self.get_value(sample, "Summary", "1 Year NAV Tracking Error")}
    Holdings: {self.get_value(sample, "Summary", "Holdings")}
    Primary: {self.get_value(sample, "Summary", "Primary")}
    Cross: {self.get_value(sample, "Summary", "Cross")}

{sample.get("Ticker", "N/A")} Performance:
    1 Day Return: {self.get_value(sample, "Performance", "1 Day Return")}
    Month-To-Date Return: {self.get_value(sample, "Performance", "Month-To-Date Return")}
    Year-To-Date Return: {self.get_value(sample, "Performance", "Year-To-Date Return")}
    1 Year Return: {self.get_value(sample, "Performance", "1 Year Return")}
    3 Years Return: {self.get_value(sample, "Performance", "3 Years Return")}
    5 Years Return: {self.get_value(sample, "Performance", "5 Years Return")}
    10 Years Return: {self.get_value(sample, "Performance", "10 Years Return")}
    12 Months Yield: {self.get_value(sample, "Performance", "12 Months Yield")}

{sample.get("Ticker", "N/A")} Liquidity:
    1 Day Volume: {self.get_value(sample, "Liquidity", "1 Day Volume")}
    Aggregated Volume: {self.get_value(sample, "Liquidity", "Aggregated Volume")}
    Aggregated Value Traded: {self.get_value(sample, "Liquidity", "Aggregated Value Traded")}
    Implied Liquidity: {self.get_value(sample, "Liquidity", "Implied Liquidity")}
    Bid Ask Spread: {self.get_value(sample, "Liquidity", "Bid Ask Spread")}
    Short Interest%: {self.get_value(sample, "Liquidity", "Short Interest%")}
    Open Interest: {self.get_value(sample, "Liquidity", "Open Interest")}

{sample.get("Ticker", "N/A")} Expense:
    Expense Ratio: {self.get_value(sample, "Expense", "Expense Ratio")}
    Fund Manager Stated Fee: {self.get_value(sample, "Expense", "Fund Manager Stated Fee")}
    Average Bid Ask Spread: {self.get_value(sample, "Expense", "Average Bid Ask Spread")}
    1 Year NAV Tracking Error: {self.get_value(sample, "Expense", "1 Year NAV Tracking Error")}
    Premium: {self.get_value(sample, "Expense", "Premium")}
    52 Weeks Average Premium: {self.get_value(sample, "Expense", "52Weeks Average Premium")}

{sample.get("Ticker", "N/A")} Flow:
    Currency: {self.get_value(sample, "Flow", "Currency, Security")}
    OAS Effective Duration: {self.get_value(sample, "Flow", "OAS Effective Duration")}
    OAS Duration Coverage Ratio: {self.get_value(sample, "Flow", "OAS Duration Coverage Ratio")}
    YAS Modified Duration: {self.get_value(sample, "Flow", "YAS Modified Duration")}
    Options Available: {self.get_value(sample, "Flow", "Options Available")}
    Payment Type: {self.get_value(sample, "Flow", "Payment Type")}

{sample.get("Ticker", "N/A")} Regulatory:
    Fund Type: {self.get_value(sample, "Regulatory", "Fund Type")}
    Structure: {self.get_value(sample, "Regulatory", "Structure")}
    Index Weight: {self.get_value(sample, "Regulatory", "Index Weight")}
    SFDR Class: {self.get_value(sample, "Regulatory", "SFDR Class.")}
    Use Derivative: {self.get_value(sample, "Regulatory", "Use Derivative")}
    Tax Form: {self.get_value(sample, "Regulatory", "Tax Form")}
    NAIC: {self.get_value(sample, "Regulatory", "NAIC")}
    UCITS: {self.get_value(sample, "Regulatory", "UCITS")}
    UK Reporting: {self.get_value(sample, "Regulatory", "UK Reporting")}
    SFC: {self.get_value(sample, "Regulatory", "SFC")}
    China: {self.get_value(sample, "Regulatory", "China")}
    Leverage: {self.get_value(sample, "Regulatory", "Leverage")}
    Inception Date: {self.get_value(sample, "Regulatory", "Inception Date")}

{sample.get("Ticker", "N/A")} Industry Exposure:
    Materials: {self.get_value(sample, "Industry", "Materials")}
    Communications: {self.get_value(sample, "Industry", "Communications")}
    Consumer Cyclical: {self.get_value(sample, "Industry", "Consumer Cyclical")}
    Consumer Non-Cyclical: {self.get_value(sample, "Industry", "Consumer Non-Cyclical")}
    Diversified: {self.get_value(sample, "Industry", "Diversified")}
    Energy: {self.get_value(sample, "Industry", "Energy")}
    Financials: {self.get_value(sample, "Industry", "Financials")}
    Industrials: {self.get_value(sample, "Industry", "Industrials")}
    Technology: {self.get_value(sample, "Industry", "Technology")}
    Utilities: {self.get_value(sample, "Industry", "Utilities")}
    Government: {self.get_value(sample, "Industry", "Government")}

{sample.get("Ticker", "N/A")} Geographical Exposure:
    North America: {self.get_value(sample, "Geography", "N.Amer.")}
    Latin America: {self.get_value(sample, "Geography", "LATAM")}
    Western Europe: {self.get_value(sample, "Geography", "West Euro")}
    Asia Pacific: {self.get_value(sample, "Geography", "APAC")}
    Eastern Europe: {self.get_value(sample, "Geography", "East Euro")}
    Africa/Middle East: {self.get_value(sample, "Geography", "Africa/Middle East")}
    Central Asia: {self.get_value(sample, "Geography", "Central Asia")}
"""
        return snippet.strip()

    def get_value(self, sample, section, key):
        if section in sample and key in sample[section]:
            value = sample[section][key]
            if value == "N":
                return "No"
            elif value == "Y":
                return "Yes"
            else:
                return value
        else:
            return "N/A"

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
        print(sample['text'])
        print()

if __name__ == '__main__':
    json_file = '/home/oleg/Documents/courses/Stanford/CS224N/FinalProject/code/FolioLLM/data/etf_data_v2.json'
    main(json_file)