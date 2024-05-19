import json
import argparse
import random


def get_value(data, keys, default="unavailable"):
    """Helper function to get a nested value from a dictionary."""
    for key in keys:
        if key in data:
            data = data[key]
            if data in ["N/A", "nan", "--", None]:
                return default
        else:
            return default
    return str(data)

def sample_etf_records(etf_records, sample_size):
    return random.sample(etf_records, sample_size)

def generate_training_data(etf_data, templates):
    training_data = []
    for etf in etf_data:
        etf_name = get_value(etf, ['Name'])
        ticker = get_value(etf, ['Ticker'])
        bbg_ticker = get_value(etf, ['BBG Ticker'])
        figi = get_value(etf, ['FIGI'])
        description = get_value(etf, ['Description'])
        fund_type = get_value(etf, ['Type'])
        domicile = get_value(etf, ['Domicile'])
        manager = get_value(etf, ['Manager'])
        fund_asset_class_focus = get_value(etf, ['Fund Asset Class Focus'])
        fund_asset_group = get_value(etf, ['Fund Asset Group'])
        fund_industry_focus = get_value(etf, ['Fund Industry Focus'])
        fund_geographical_focus = get_value(etf, ['Fund Geographical Focus'])
        fund_objective = get_value(etf, ['Fund Objective'])
        economic_association = get_value(etf, ['Economic Association'])
        fund_strategy = get_value(etf, ['Fund Strategy'])
        fund_market_cap_focus = get_value(etf, ['Fund Market Cap Focus'])
        fund_style = get_value(etf, ['Fund Style'])
        tot_ret_ytd = get_value(etf, ['Tot Ret Ytd'])
        total_return_1y = get_value(etf, ['Tot Ret 1Y'])
        class_assets = get_value(etf, ['Summary', 'Class Assets (MLN USD)'])
        fund_assets = get_value(etf, ['Summary', 'Fund Assets (MLN USD)'])
        expense_ratio = get_value(etf, ['Summary', 'Expense Ratio'])
        year_to_date_return = get_value(etf, ['Summary', 'Year-To-Date Return'])
        yield_12m = get_value(etf, ['Summary', '12Months Yield'])
        volume_30d = get_value(etf, ['Summary', '30Days Volatility'])
        ytd_flow = get_value(etf, ['Summary', 'Year-To-Date Flow'])
        flow_1m = get_value(etf, ['Summary', '1Month Flow'])
        nav_trk_error = get_value(etf, ['Summary', '1 Year NAV Tracking Error'])
        holdings = get_value(etf, ['Summary', 'Holdings'])
        primary = "primary" if get_value(etf, ['Summary', 'Primary']) == "Y" else "cross"
        return_1d = get_value(etf, ['Performance', '1 Day Return'])
        return_mtd = get_value(etf, ['Performance', 'Month-To-Date Return'])
        return_ytd = get_value(etf, ['Performance', 'Year-To-Date Return'])
        return_3y = get_value(etf, ['Performance', '3 Years Return'])
        return_5y = get_value(etf, ['Performance', '5 Years Return'])
        return_10y = get_value(etf, ['Performance', '10 Years Return'])
        volume_1d = get_value(etf, ['Liquidity', '1 Day Volume'])
        aggregated_volume = get_value(etf, ['Liquidity', 'Aggregated Volume'])
        aggregated_value_traded = get_value(etf, ['Liquidity', 'Aggregated Value Traded'])
        short_interest = get_value(etf, ['Liquidity', 'Short Interest%'])
        open_interest = get_value(etf, ['Liquidity', 'Open Interest'])
        total_value_traded = get_value(etf, ['Liquidity', 'Aggregated Value Traded'])
        bid_ask_spread = get_value(etf, ['Liquidity', 'Bid Ask Spread'])
        implied_liquidity = get_value(etf, ['Liquidity', 'Implied Liquidity'])
        inception_date = get_value(etf, ['Regulatory', 'Inception Date'])
        inception_year = inception_date.split('/')[-1] if inception_date != "unavailable" else "unavailable"
        use_derivative = "Yes" if get_value(etf, ['Regulatory', 'Use Derivative']) == "Y" else "No"
        payment_type = get_value(etf, ['Flow', 'Payment Type'])
        leverage = "uses leverage" if get_value(etf, ['Regulatory', 'Leverage']) == "Y" else "does not use leverage"
        structure = get_value(etf, ['Regulatory', 'Structure'])
        avg_bid_ask_spread = get_value(etf, ['Expense', 'Average Bid Ask Spread'])

        # Log missing keys
        # missing_keys = []
        # required_keys = {
        #     'etf_name': etf_name, 'ticker': ticker, 'bbg_ticker': bbg_ticker, 'figi': figi, 'description': description,
        #     'fund_type': fund_type, 'domicile': domicile, 'manager': manager, 'fund_asset_class_focus': fund_asset_class_focus,
        #     'fund_asset_group': fund_asset_group, 'fund_industry_focus': fund_industry_focus, 'fund_geographical_focus': fund_geographical_focus,
        #     'fund_objective': fund_objective, 'economic_association': economic_association, 'fund_strategy': fund_strategy,
        #     'fund_market_cap_focus': fund_market_cap_focus, 'fund_style': fund_style, 'tot_ret_ytd': tot_ret_ytd,
        #     'total_return_1y': total_return_1y, 'class_assets': class_assets, 'fund_assets': fund_assets,
        #     'expense_ratio': expense_ratio, 'year_to_date_return': year_to_date_return, 'yield_12m': yield_12m,
        #     'volume_30d': volume_30d, 'ytd_flow': ytd_flow, 'flow_1m': flow_1m, 'nav_trk_error': nav_trk_error,
        #     'holdings': holdings, 'primary': primary, 'return_1d': return_1d, 'return_mtd': return_mtd,
        #     'return_3y': return_3y, 'return_5y': return_5y, 'return_10y': return_10y, 'short_interest': short_interest,
        #     'open_interest': open_interest, 'total_value_traded': total_value_traded, 'bid_ask_spread': bid_ask_spread,
        #     'implied_liquidity': implied_liquidity, 'inception_date': inception_date, 'inception_year': inception_year,
        #     'use_derivative': use_derivative, 'payment_type': payment_type, 'leverage': leverage, 'structure': structure,
        #     'avg_bid_ask_spread': avg_bid_ask_spread
        # }

        # for key, value in required_keys.items():
        #     if value == "unavailable":
        #         missing_keys.append(key)
        #
        # if missing_keys:
        #     print(f"Missing keys for ETF {etf_name} ({ticker}): {', '.join(missing_keys)}")

        for template in templates:
            try:
                prompt = template['prompt'].replace("[ETF Name]", etf_name)
                response = template['response'].format(
                    etf_name=etf_name,
                    ticker=ticker,
                    bbg_ticker=bbg_ticker,
                    figi=figi,
                    description=description,
                    fund_type=fund_type,
                    domicile=domicile,
                    manager=manager,
                    asset_class_focus=fund_asset_class_focus,
                    fund_asset_group=fund_asset_group,
                    fund_industry_focus=fund_industry_focus,
                    fund_geographical_focus=fund_geographical_focus,
                    fund_objective=fund_objective,
                    economic_association=economic_association,
                    fund_strategy=fund_strategy,
                    fund_market_cap_focus=fund_market_cap_focus,
                    fund_style=fund_style,
                    tot_ret_ytd=tot_ret_ytd,
                    total_return_1y=total_return_1y,
                    class_assets=class_assets,
                    fund_assets=fund_assets,
                    expense_ratio=expense_ratio,
                    ytd_return=return_ytd,
                    yield_12m=yield_12m,
                    volume_30d=volume_30d,
                    ytd_flow=ytd_flow,
                    flow_1m=flow_1m,
                    nav_trk_error=nav_trk_error,
                    holdings=holdings,
                    primary=primary,
                    return_1d=return_1d,
                    return_mtd=return_mtd,
                    return_3y=return_3y,
                    return_5y=return_5y,
                    return_10y=return_10y,
                    short_interest=short_interest,
                    open_interest=open_interest,
                    total_value_traded=total_value_traded,
                    bid_ask_spread=bid_ask_spread,
                    implied_liquidity=implied_liquidity,
                    inception_date=inception_date,
                    inception_year=inception_year,
                    use_derivative=use_derivative,
                    payment_type=payment_type,
                    leverage=leverage,
                    structure=structure,
                    avg_bid_ask_spread=avg_bid_ask_spread,
                    aggregated_value_traded=aggregated_value_traded,
                )
                training_data.append({"prompt": prompt, "response": response})
            except KeyError as e:
                print(f"Error formatting response for ETF {etf_name} ({ticker}): missing key {e}")

    return training_data

def main(etf_data_file, template_data_file, output_file, sample_size):
    # Load ETF data
    with open(etf_data_file, 'r') as f:
        etf_data = json.load(f)

    # Load templates
    with open(template_data_file, 'r') as f:
        templates = json.load(f)

    # Sample 10-20% of the ETF records
    sample_size = int(len(etf_data) * sample_size)
    sampled_records = sample_etf_records(etf_data, sample_size)

    # Generate training data
    training_data = generate_training_data(sampled_records, templates)

    # Save the training data
    with open(output_file, 'w') as f:
        json.dump(training_data, f, indent=4)

    print(f"Prompt/response training data generated and saved to {output_file}. \nSample size: {sample_size}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training data for ETF advisor chatbot")
    parser.add_argument("sample_size", type=float, help="Sample size (e.g. 0.2)")
    parser.add_argument("etf_data_file", help="Path to the ETF data file")
    parser.add_argument("template_data_file", help="Path to the template data file")
    parser.add_argument("output_file", help="Path to the output file where training data will be saved")

    args = parser.parse_args()
    main(args.etf_data_file, args.template_data_file, args.output_file, args.sample_size)
