

def translate_etf_entry(etf):
    """
    Translate the keys in a single ETF entry according to specific rules.

    Args:
        etf (dict): A dictionary containing an ETF entry.

    Returns:
        dict: A dictionary with translated keys.
    """
    return  {
    'ticker': get_value(etf, ['Ticker']),
    'bbg_ticker': get_value(etf, ['BBG Ticker']),
    'etf_name': get_value(etf, ['Name']),
    'description': get_value(etf, ['Description']),
    'fund_type': get_value(etf, ['Type']),
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

    'class_assets': get_value(etf, ['Summary', 'Class Assets (MLN USD)']),
    'fund_assets': get_value(etf, ['Summary', 'Fund Assets (MLN USD)']),
    'expense_ratio': get_value(etf, ['Summary', 'Expense Ratio']),
    'year_to_date_return': get_value(etf, ['Summary', 'Year-To-Date Return']),
    'summary_12_months_yield': get_value(etf, ['Summary', '12Months Yield']),
    'volume_30d': get_value(etf, ['Summary', '30Days Volatility']),
    'ytd_flow': get_value(etf, ['Summary', 'Year-To-Date Flow']),
    'flow_1m': get_value(etf, ['Summary', '1Month Flow']),
    'nav_trk_error': get_value(etf, ['Summary', '1 Year NAV Tracking Error']),
    'holdings': get_value(etf, ['Summary', 'Holdings']),
    'primary': get_value(etf, ['Summary', 'Primary']),
    'cross': get_value(etf, ['Summary', 'Cross']),

    'return_1d': get_value(etf, ['Performance', '1 Day Return']),
    'return_mtd': get_value(etf, ['Performance', 'Month-To-Date Return']),
    'ytd_return': get_value(etf, ['Performance', 'Year-To-Date Return']),
    'performance_1_year_return': get_value(etf, ['Performance', '1 Year Return']),
    'return_3y': get_value(etf, ['Performance', '3 Years Return']),
    'performance_5_years_return': get_value(etf, ['Performance', '5 Years Return']),
    'performance_10_years_return': get_value(etf, ['Performance', '10 Years Return']),
    'performance_12_months_yield': get_value(etf, ['Performance', '12 Months Yield']),

    'volume_1d': get_value(etf, ['Liquidity', '1 Day Volume']),
    'aggregated_volume': get_value(etf, ['Liquidity', 'Aggregated Volume']),
    'total_value_traded': get_value(etf, ['Liquidity', 'Aggregated Value Traded']),
    'liquidity_implied_liquidity': get_value(etf, ['Liquidity', 'Implied Liquidity']),
    'bid_ask_spread': get_value(etf, ['Liquidity', 'Bid Ask Spread']),
    'liquidity_short_interest': get_value(etf, ['Liquidity', 'Short Interest%']),
    'liquidity_open_interest': get_value(etf, ['Liquidity', 'Open Interest']),

    'expense_ratio': get_value(etf, ['Expense', 'Expense Ratio']),
    'expense_fund_manager_stated_fee': get_value(etf, ['Expense', 'Fund Manager Stated Fee']),
    'avg_bid_ask_spread': get_value(etf, ['Expense', 'Average Bid Ask Spread']),
    'nav_trk_error': get_value(etf, ['Expense', '1 Year NAV Tracking Error']),
    'premium': get_value(etf, ['Expense', 'Premium']),
    'premium_52w': get_value(etf, ['Expense', '52Weeks Average Premium']),

    'currency': get_value(etf, ['Flow', 'Currency, Security']),
    'oas_effective_duration': get_value(etf, ['Flow', 'OAS Effective Duration']),
    'oas_duration_coverage_ratio': get_value(etf, ['Flow', 'OAS Duration Coverage Ratio']),
    'yas_modified_duration': get_value(etf, ['Flow', 'YAS Modified Duration']),
    'options_available': get_value(etf, ['Flow', 'Options Available']),
    'payment_type': get_value(etf, ['Flow', 'Payment Type']),

    'fund_type': get_value(etf, ['Regulatory', 'Fund Type']),
    'structure': get_value(etf, ['Regulatory', 'Structure']),
    'index_weight': get_value(etf, ['Regulatory', 'Index Weight']),
    'sfdr_class': get_value(etf, ['Regulatory', 'SFDR Class.']),
    'use_derivative': get_value(etf, ['Regulatory', 'Use Derivative']),
    'tax_form': get_value(etf, ['Regulatory', 'Tax Form']),
    'naic': get_value(etf, ['Regulatory', 'NAIC']),
    'ucits': get_value(etf, ['Regulatory', 'UCITS']),
    'uk_reporting': get_value(etf, ['Regulatory', 'UK Reporting']),
    'sfc': get_value(etf, ['Regulatory', 'SFC']),
    'china': get_value(etf, ['Regulatory', 'China']),
    'leverage': get_value(etf, ['Regulatory', 'Leverage']),
    'inception_date': get_value(etf, ['Regulatory', 'Inception Date']),

    'materials': get_value(etf, ['Industry', 'Materials']),
    'communications': get_value(etf, ['Industry', 'Communications']),
    'consumer_cyclical': get_value(etf, ['Industry', 'Consumer Cyclical']),
    'consumer_non_cyclical': get_value(etf, ['Industry', 'Consumer Non-Cyclical']),
    'diversified': get_value(etf, ['Industry', 'Diversified']),
    'energy': get_value(etf, ['Industry', 'Energy']),
    'financials': get_value(etf, ['Industry', 'Financials']),
    'industrials': get_value(etf, ['Industry', 'Industrials']),
    'technology': get_value(etf, ['Industry', 'Technology']),
    'utilities': get_value(etf, ['Industry', 'Utilities']),
    'government': get_value(etf, ['Industry', 'Government']),

    'north_america': get_value(etf, ['Geography', 'N.Amer.']),
    'latin_america': get_value(etf, ['Geography', 'LATAM']),
    'western_europe': get_value(etf, ['Geography', 'West Euro']),
    'asia_pacific': get_value(etf, ['Geography', 'APAC']),
    'eastern_europe': get_value(etf, ['Geography', 'East Euro']),
    'africa_middle_east': get_value(etf, ['Geography', 'Africa/Middle East']),
    'central_asia': get_value(etf, ['Geography', 'Central Asia'])
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
