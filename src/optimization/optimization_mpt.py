import pandas as pd
import numpy as np
import datetime as dt
from scipy.optimize import minimize
import os

test_tickers = ['SPY US Equity', 'IVV US Equity', 'VO US Equity', '510050 CH Equity']

opt_instance_test = None
opt_instance_main = None

current_file_path = os.path.abspath(os.path.dirname(__file__))

# Resolve the absolute path of the relative file
abs_path = lambda path: os.path.abspath(os.path.join(current_file_path, path))

main_prices_file = abs_path('../../data/etf_prices.xlsx')
test_prices_file = abs_path('../../data/test_prices.xlsx')


class PortfolioOptimizer:
    def __init__(self, tickers, data_file, risk_free_rate=0.05):
        self.tickers = tickers
        self.data_file = data_file
        self.risk_free_rate = risk_free_rate
        self.data, self.benchmark = self.load_data()
        self.returns, self.benchmark_returns = self.calculate_returns()

    def load_data(self):
        # Load the CSV file containing daily returns
        data = pd.read_excel(self.data_file, sheet_name='Combined', index_col='Date', parse_dates=True)
        # Convert index to datetime if not already in datetime format
        data.index = pd.to_datetime(data.index).normalize()
        benchmark = data['SPY US Equity']
        # Ensure only the tickers we are interested in are selected
        ticker_columns = [ticker for ticker in self.tickers]
        data = data[ticker_columns]
        data = data.fillna(method='ffill')

        # Rename columns to match tickers without ' Equity' suffix
        data.columns = self.tickers
        return data, benchmark

    def calculate_returns(self):
        # Select one year period of data
        #end_date = self.data.index.max()
        #start_date = end_date - pd.DateOffset(years=1)
        #one_year_data = self.data.loc[end_date:start_date]
        one_year_data = self.data

        # Ensure data is sorted by date in ascending order
        one_year_data = one_year_data.sort_index()

        # Calculate daily returns
        returns = one_year_data.pct_change().dropna()

        # Check if data for the period is available
        if one_year_data.empty:
            raise ValueError("No data available for the selected one-year period.")

        benchmark_returns = self.benchmark.pct_change().dropna()

        return returns, benchmark_returns

    def compute_statistics(self):
        # Calculate mean return, variance, and standard deviation
        mean_returns = self.returns.mean()
        cov_matrix = self.returns.cov()
        std_devs = self.returns.std()
        return mean_returns, cov_matrix, std_devs

    def mean_variance_optimization(self):
        mean_returns, cov_matrix, _ = self.compute_statistics()
        num_assets = len(self.tickers)

        def portfolio_return(weights):
            return np.sum(weights * mean_returns)

        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        def negative_sharpe_ratio(weights):
            return -(portfolio_return(weights) - self.risk_free_rate / 252) / portfolio_volatility(weights)

        # Constraints and bounds
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = num_assets * [1. / num_assets, ]

        # Optimize the portfolio
        result = minimize(negative_sharpe_ratio, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

        return result.x

    def get_portfolio_details(self):
        weights = self.mean_variance_optimization()
        mean_returns, cov_matrix, _ = self.compute_statistics()

        daily_portfolio_return = np.sum(weights * mean_returns)
        daily_portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        # Annualize return and volatility
        annualized_return = (1 + daily_portfolio_return) ** 252 - 1
        annualized_volatility = daily_portfolio_volatility * np.sqrt(252)

        # Calculate Sharpe ratio
        annual_risk_free_rate = self.risk_free_rate
        sharpe_ratio = (annualized_return - annual_risk_free_rate) / annualized_volatility

        # Calculate Sortino ratio
        portfolio_returns = self.returns.dot(weights)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = np.std(downside_returns)
        annual_downside_deviation = downside_deviation * np.sqrt(252)
        sortino_ratio = (annualized_return - annual_risk_free_rate) / annual_downside_deviation

        # Calculate Information ratio
        benchmark_return = self.benchmark_returns.mean()  # Assuming 'SPY US' as the benchmark
        tracking_error = np.sqrt(np.mean((self.returns.dot(weights) - self.benchmark_returns) ** 2)) * np.sqrt(252)
        information_ratio = (annualized_return - (benchmark_return * 252)) / tracking_error

        # Convert weights to percentages
        weights_percent = weights * 100

        # Create a DataFrame for weights
        weights_df = pd.DataFrame({'Ticker': self.tickers, 'Weight (%)': weights_percent})

        # Format weights to display as percentages
        pd.options.display.float_format = '{:.2f}'.format

        return {
            'weights_table': weights_df,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'information_ratio': information_ratio
        }

def optimizer(tickers=test_tickers, main=True):
    global opt_instance_main, opt_instance_test
    risk_free_rate = 0.05
    if main:
        if not opt_instance_main:
            print("New instance of optim pd df")
            opt_instance_main = PortfolioOptimizer(
                tickers, main_prices_file, risk_free_rate
            )
            print(opt_instance_main)
        else: print("Using saved optim pd df")
        optimizer = opt_instance_main
    else:
        if not opt_instance_test:
            print("New instance of optim pd df")
            opt_instance_test = PortfolioOptimizer(
                tickers, test_prices_file, risk_free_rate
            )
            print(opt_instance_test)
        else: print("Using saved optim pd df")
        optimizer = opt_instance_test   

    # wtf why details are not fetched
    portfolio_details = optimizer.get_portfolio_details()

    # in future we will need to use something other than {portfolio_details[...]}
    template = f"""
    {portfolio_details['weights_table']}
    Annualized Return: {portfolio_details['annualized_return']:.2%}
    Annualized Volatility: {portfolio_details['annualized_volatility']:.2%}
    Sharpe Ratio: {portfolio_details['sharpe_ratio']:.2f}
    Sortino Ratio: {portfolio_details['sortino_ratio']:.2f}
    Information Ratio: {portfolio_details['information_ratio']:.2f}"""
    return template

