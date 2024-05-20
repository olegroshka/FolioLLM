def translate_etf_entry(etf):
    """
    Translate the keys in a single ETF entry according to specific rules.

    Args:
        etf (dict): A dictionary containing an ETF entry.

    Returns:
        dict: A dictionary with translated keys.
    """
    return {
        'etf_name': get_value(etf, ['Name']),
        'ticker': get_value(etf, ['Ticker']),
        'bbg_ticker': get_value(etf, ['BBG Ticker']),
        'figi': get_value(etf, ['FIGI']),
        'description': get_value(etf, ['Description']),
        'fund_type': get_value(etf, ['Type']),
        'domicile': get_value(etf, ['Domicile']),
        'manager': get_value(etf, ['Manager']),
        'asset_class_focus': get_value(etf, ['Fund Asset Class Focus']),
        'fund_asset_group': get_value(etf, ['Fund Asset Group']),
        'fund_industry_focus': get_value(etf, ['Fund Industry Focus']),
        'fund_geographical_focus': get_value(etf, ['Fund Geographical Focus']),
        'fund_objective': get_value(etf, ['Fund Objective']),
        'economic_association': get_value(etf, ['Economic Association']),
        'fund_strategy': get_value(etf, ['Fund Strategy']),
        'fund_market_cap_focus': get_value(etf, ['Fund Market Cap Focus']),
        'fund_style': get_value(etf, ['Fund Style']),
        'tot_ret_ytd': get_value(etf, ['Tot Ret Ytd']),
        'total_return_1y': get_value(etf, ['Tot Ret 1Y']),
        'class_assets': get_value(etf, ['Summary', 'Class Assets (MLN USD)']),
        'fund_assets': get_value(etf, ['Summary', 'Fund Assets (MLN USD)']),
        'expense_ratio': get_value(etf, ['Summary', 'Expense Ratio']),
        'year_to_date_return': get_value(etf, ['Summary', 'Year-To-Date Return']),
        'yield_12m': get_value(etf, ['Summary', '12Months Yield']),
        'volume_30d': get_value(etf, ['Summary', '30Days Volatility']),
        'ytd_flow': get_value(etf, ['Summary', 'Year-To-Date Flow']),
        'flow_1m': get_value(etf, ['Summary', '1Month Flow']),
        'nav_trk_error': get_value(etf, ['Summary', '1 Year NAV Tracking Error']),
        'holdings': get_value(etf, ['Summary', 'Holdings']),
        'primary': "primary" if get_value(etf, ['Summary', 'Primary']) == "Y" else "cross",
        'return_1d': get_value(etf, ['Performance', '1 Day Return']),
        'return_mtd': get_value(etf, ['Performance', 'Month-To-Date Return']),
        'ytd_return': get_value(etf, ['Performance', 'Year-To-Date Return']),
        'return_3y': get_value(etf, ['Performance', '3 Years Return']),
        'return_5y': get_value(etf, ['Performance', '5 Years Return']),
        'return_10y': get_value(etf, ['Performance', '10 Years Return']),
        'volume_1d': get_value(etf, ['Liquidity', '1 Day Volume']),
        'aggregated_volume': get_value(etf, ['Liquidity', 'Aggregated Volume']),
        'aggregated_value_traded': get_value(etf, ['Liquidity', 'Aggregated Value Traded']),
        'short_interest': get_value(etf, ['Liquidity', 'Short Interest%']),
        'open_interest': get_value(etf, ['Liquidity', 'Open Interest']),
        'total_value_traded': get_value(etf, ['Liquidity', 'Aggregated Value Traded']),
        'bid_ask_spread': get_value(etf, ['Liquidity', 'Bid Ask Spread']),
        'implied_liquidity': get_value(etf, ['Liquidity', 'Implied Liquidity']),
        'inception_date': get_value(etf, ['Regulatory', 'Inception Date']),
        'inception_year': get_value(etf, ['Regulatory', 'Inception Date']).split('/')[-1] if get_value(etf, ['Regulatory', 'Inception Date']) != "unavailable" else "unavailable",
        'use_derivative': "Yes" if get_value(etf, ['Regulatory', 'Use Derivative']) == "Y" else "No",
        'payment_type': get_value(etf, ['Flow', 'Payment Type']),
        'leverage': "uses leverage" if get_value(etf, ['Regulatory', 'Leverage']) == "Y" else "does not use leverage",
        'structure': get_value(etf, ['Regulatory', 'Structure']),
        'avg_bid_ask_spread': get_value(etf, ['Expense', 'Average Bid Ask Spread'])
    }

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
