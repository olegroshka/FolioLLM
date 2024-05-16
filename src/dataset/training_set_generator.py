import json
import argparse

def get_value(data, keys, default="unavailable"):
    """Helper function to get a nested value from a dictionary."""
    for key in keys:
        if key in data:
            data = data[key]
            if data in ["N/A", "nan", "--", None]:
                return default
        else:
            return default
    return data

def generate_training_data(etf_data, templates):
    training_data = []
    for etf in etf_data:
        etf_name = get_value(etf, ['Name'])
        ticker = get_value(etf, ['Ticker'])
        manager = get_value(etf, ['Manager'])
        ytd_return = get_value(etf, ['Tot Ret Ytd'])
        expense_ratio = get_value(etf, ['Expense', 'Expense Ratio'])
        total_return_1y = get_value(etf, ['Tot Ret 1Y'])
        holdings = get_value(etf, ['Summary', 'Holdings'])
        volume_30d = get_value(etf, ['Summary', '30D Vol'])
        asset_class_focus = get_value(etf, ['Descriptive', 'Fund Asset Class Focus'])
        class_assets = get_value(etf, ['Summary', 'Class Assets (MLN USD)'])
        fund_assets = get_value(etf, ['Summary', 'Fund Assets (MLN USD)'])
        yield_12m = get_value(etf, ['Summary', '12M Yld'])
        flow_1m = get_value(etf, ['Summary', '1M Flow'])
        ytd_flow = get_value(etf, ['Summary', 'YTD Flow'])
        primary = "primary" if get_value(etf, ['Summary', 'Primary']) == "Y" else "cross"
        nav_trk_error = get_value(etf, ['Summary', '1 Yr NAV Trk Error'])
        inception_date = get_value(etf, ['Regulatory', 'Inception Date'])
        inception_year = inception_date.split('/')[-1] if inception_date != "unavailable" else "unavailable"
        use_derivative = "Yes" if get_value(etf, ['Regulatory', 'Use Derivative']) == "Y" else "No"
        payment_type = get_value(etf, ['Flow', 'Payment Type'])
        total_value_traded = get_value(etf, ['Liquidity', 'Aggregated Value Traded'])
        bid_ask_spread = get_value(etf, ['Liquidity', 'Bid Ask Spread'])
        short_interest = get_value(etf, ['Liquidity', 'Short Interest%'])
        open_interest = get_value(etf, ['Liquidity', 'Open Interest'])
        economic_association = get_value(etf, ['Descriptive', 'Economic Association'])
        industry_focus = get_value(etf, ['Descriptive', 'Fund Industry Focus'])
        return_1d = get_value(etf, ['Performance', '1D Return'])
        return_mtd = get_value(etf, ['Performance', 'MTD Return'])
        avg_bid_ask_spread = get_value(etf, ['Expense', 'Avg Bid Ask Spread'])
        structure = get_value(etf, ['Regulatory', 'Structure'])
        leverage = "uses leverage" if get_value(etf, ['Regulatory', 'Leverage']) == "Y" else "does not use leverage"
        fund_strategy = get_value(etf, ['Descriptive', 'Fund Strategy'])
        fund_geographical_focus = get_value(etf, ['Descriptive', 'Fund Geographical Focus'])
        tot_ret_ytd = get_value(etf, ['Tot Ret Ytd'])
        aggregated_value_traded = get_value(etf, ['Liquidity', 'Aggregated Value Traded'])

        # Log missing keys
        missing_keys = []
        required_keys = {
            'etf_name': etf_name, 'ticker': ticker, 'manager': manager, 'ytd_return': ytd_return,
            'expense_ratio': expense_ratio, 'total_return_1y': total_return_1y, 'holdings': holdings,
            'volume_30d': volume_30d, 'asset_class_focus': asset_class_focus, 'class_assets': class_assets,
            'fund_assets': fund_assets, 'yield_12m': yield_12m, 'flow_1m': flow_1m, 'ytd_flow': ytd_flow,
            'primary': primary, 'nav_trk_error': nav_trk_error, 'inception_date': inception_date,
            'inception_year': inception_year, 'use_derivative': use_derivative, 'payment_type': payment_type,
            'total_value_traded': total_value_traded, 'bid_ask_spread': bid_ask_spread, 'short_interest': short_interest,
            'open_interest': open_interest, 'economic_association': economic_association, 'industry_focus': industry_focus,
            'return_1d': return_1d, 'return_mtd': return_mtd, 'avg_bid_ask_spread': avg_bid_ask_spread, 'structure': structure,
            'leverage': leverage, 'fund_strategy': fund_strategy, 'fund_geographical_focus': fund_geographical_focus,
            'tot_ret_ytd': tot_ret_ytd, 'aggregated_value_traded': aggregated_value_traded
        }

        for key, value in required_keys.items():
            if value == "unavailable":
                missing_keys.append(key)

        if missing_keys:
            print(f"Missing keys for ETF {etf_name} ({ticker}): {', '.join(missing_keys)}")

        for template in templates:
            try:
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
                    inception_year=inception_year,
                    use_derivative=use_derivative,
                    payment_type=payment_type,
                    total_value_traded=total_value_traded,
                    bid_ask_spread=bid_ask_spread,
                    short_interest=short_interest,
                    open_interest=open_interest,
                    economic_association=economic_association,
                    industry_focus=industry_focus,
                    return_1d=return_1d,
                    return_mtd=return_mtd,
                    avg_bid_ask_spread=avg_bid_ask_spread,
                    structure=structure,
                    leverage=leverage,
                    fund_strategy=fund_strategy,
                    fund_geographical_focus=fund_geographical_focus,
                    tot_ret_ytd=tot_ret_ytd,
                    aggregated_value_traded=aggregated_value_traded
                )
                training_data.append({"prompt": prompt, "response": response})
            except KeyError as e:
                print(f"Error formatting response for ETF {etf_name} ({ticker}): missing key {e}")

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
