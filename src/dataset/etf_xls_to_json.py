import argparse
import json
import pandas as pd

def process_xls_file(file_path):
    # Read the XLSX file and select the "excelexportfsrcresults" tab
    df_results = pd.read_excel(file_path, sheet_name="excelexportfsrcresults")
    #print(df_results.columns)

    # Extract the required fields from the "excelexportfsrcresults" tab
    required_fields = ["Ticker\n", "Tot Ret Ytd\n", "Tot Ret 1Y\n", "Fund Asset Class Focus"]
    required_fields = [col.strip() for col in required_fields]
    data_results = df_results[required_fields].to_dict(orient="records")

    # Read the XLS file and select the "Summary" tab
    df_summary = pd.read_excel(file_path, sheet_name="Summary")

    #print("df_summary.columns", df_summary.columns)

    # Create a dictionary mapping ticker to summary data
    summary_fields = ["Name", "Ticker", "Class Assets (MLN USD)", "Fund Assets (MLN USD)", "Expense Ratio",
                      "YTD Return", "12M Yld", "30D Vol", "YTD Flow", "1M Flow", "1 Yr NAV Trk Error",
                      "Holdings", "Primary", "Cross"]

    summary_data = {}
    for _, row in df_summary[summary_fields].iterrows():
        if pd.isna(row["Ticker"]):
            print(f"Warning: Skipping row with missing Ticker: \n{row}")
            continue
        ticker = row["Ticker"].split(" ")[0]
        summary_data[ticker] = row

    # Merge the data from both tabs based on ticker
    for item in data_results:
        ticker_name = item["Ticker"]
        ticker = ticker_name.split(" ")[0]  # Extract the actual ticker
        if ticker in summary_data:
            item["ETFTickerName"] = ticker_name
            item["ETFTicker"] = ticker
            item["Summary"] = summary_data[ticker].to_dict()
        else:
            item["Ticker"] = ticker

    return data_results

def save_to_json(data, output_file):
    with open(output_file, "w") as json_file:
        json.dump(data, json_file, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Process XLS file and create JSON dataset")
    parser.add_argument("xls_file", help="Path to the input XLS file")
    parser.add_argument("output_file", help="Path to the output JSON file")
    args = parser.parse_args()

    data = process_xls_file(args.xls_file)
    save_to_json(data, args.output_file)

    print(f"JSON dataset created: {args.output_file}")

if __name__ == "__main__":
    main()