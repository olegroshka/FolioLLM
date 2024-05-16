import json
import argparse


def generate_training_data(etf_data, templates):
    training_data = []
    for etf in etf_data:
        try:
            etf_name = etf['Name']
            ticker = etf['Ticker']
            manager = etf['Manager']
            ytd_return = etf['Tot Ret Ytd']
            expense_ratio = etf['Expense']['Expense Ratio']
            total_return_1y = etf['Tot Ret 1Y']
            holdings = etf['Summary']['Holdings']
            volume_30d = etf['Summary']['30D Vol']
            asset_class_focus = etf['Descriptive']['Fund Asset Class Focus']
            class_assets = etf['Summary']['Class Assets (MLN USD)']
            fund_assets = etf['Summary']['Fund Assets (MLN USD)']
            yield_12m = etf['Summary']['12M Yld']
            flow_1m = etf['Summary']['1M Flow']
            ytd_flow = etf['Summary']['YTD Flow']
            primary = "primary" if etf['Summary']['Primary'] == "Y" else "cross"
            nav_trk_error = etf['Summary']['1 Yr NAV Trk Error']
            inception_date = etf['Regulatory']['Inception Date']
            use_derivative = "Yes" if etf['Regulatory']['Use Derivative'] == "Y" else "No"
            payment_type = etf['Flow']['Payment Type']
            total_value_traded = etf['Liquidity']['Aggregated Value Traded']
            bid_ask_spread = etf['Liquidity']['Bid Ask Spread']
            short_interest = etf['Liquidity']['Short Interest%']
            open_interest = etf['Liquidity']['Open Interest']
            economic_association = etf['Descriptive']['Economic Association']
            industry_focus = etf['Descriptive']['Fund Industry Focus']
            return_1d = etf['Performance']['1D Return']
            return_mtd = etf['Performance']['MTD Return']

            for template in templates:
                prompt = template['prompt'].replace("[ETF Name]", etf_name)
                response = template['response'].format(
                    etf_name=etf_name,
                    ticker=ticker,
                    manager=manager,
                    ytd_return=ytd_return,
                    expense_ratio=expense_ratio,
                    total_return_1y=total_return_1y,
                    holdings=holdings,
                    volume_30d=volume_30d,
                    asset_class_focus=asset_class_focus,
                    class_assets=class_assets,
                    fund_assets=fund_assets,
                    yield_12m=yield_12m,
                    flow_1m=flow_1m,
                    ytd_flow=ytd_flow,
                    primary=primary,
                    nav_trk_error=nav_trk_error,
                    inception_date=inception_date,
                    use_derivative=use_derivative,
                    payment_type=payment_type,
                    total_value_traded=total_value_traded,
                    bid_ask_spread=bid_ask_spread,
                    short_interest=short_interest,
                    open_interest=open_interest,
                    economic_association=economic_association,
                    industry_focus=industry_focus,
                    return_1d=return_1d,
                    return_mtd=return_mtd
                )
                training_data.append({"prompt": prompt, "response": response})

        except KeyError as e:
            print(f"Missing key in ETF data: {e}")
            continue

    return training_data


def main(etf_data_file, template_data_file, output_file):
    # Load ETF data
    with open(etf_data_file, 'r') as f:
        etf_data = json.load(f)

    # Load templates
    with open(template_data_file, 'r') as f:
        templates = json.load(f)

    # Generate training data
    training_data = generate_training_data(etf_data, templates)

    # Save the training data
    with open(output_file, 'w') as f:
        json.dump(training_data, f, indent=4)

    print(f"Training data generated and saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training data for ETF advisor chatbot")
    parser.add_argument("etf_data_file", help="Path to the ETF data file")
    parser.add_argument("template_data_file", help="Path to the template data file")
    parser.add_argument("output_file", help="Path to the output file where training data will be saved")

    args = parser.parse_args()
    main(args.etf_data_file, args.template_data_file, args.output_file)
