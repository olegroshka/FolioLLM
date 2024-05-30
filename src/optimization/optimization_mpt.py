import pandas as pd
import numpy as np
import datetime as dt
from scipy.optimize import minimize


class PortfolioOptimizer:
    def __init__(self, tickers, data_file, risk_free_rate=0.05):
        self.tickers = tickers
        self.data_file = data_file
        self.risk_free_rate = risk_free_rate
        self.data = self.load_data()
        self.returns = self.calculate_returns()

    def load_data(self):
        # Load the CSV file containing daily returns
        data = pd.read_excel(self.data_file, sheet_name='Combined', header=1, index_col='Date', parse_dates=True)
        # Convert index to datetime if not already in datetime format
        data.index = pd.to_datetime(data.index).normalize()

        # Ensure only the tickers we are interested in are selected
        ticker_columns = [ticker for ticker in self.tickers]
        data = data[ticker_columns]
        # Rename columns to match tickers without ' Equity' suffix
        data.columns = self.tickers
        return data

    def calculate_returns(self):
        # Select one year period of data
        end_date = self.data.index.max()
        start_date = end_date - pd.DateOffset(years=1)
        one_year_data = self.data.loc[end_date:start_date]

        # Ensure data is sorted by date in ascending order
        one_year_data = one_year_data.sort_index()

        # Calculate daily returns
        returns = one_year_data.pct_change().dropna()

        # Check if data for the period is available
        if one_year_data.empty:
            raise ValueError("No data available for the selected one-year period.")


        return returns

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
        benchmark_return = mean_returns['SPY US']  # Assuming 'SPY US' as the benchmark
        tracking_error = np.sqrt(np.mean((self.returns.dot(weights) - self.returns['SPY US']) ** 2)) * np.sqrt(252)
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

def main_optimizer_mpt(tickers, ):
    tickers = ['SPY US', 'IVY US', 'VO US', '510050 CH']  # to be modified
    data_file = '.../data/etf_prices.xlsx'  # Path to the uploaded Excel file
    risk_free_rate = 0.05  # 5% annual risk-free rate

    optimizer = PortfolioOptimizer(tickers, data_file, risk_free_rate)
    portfolio_details = optimizer.get_portfolio_details()

    print("Weights:")
    print(portfolio_details['weights_table'])
    print(f"Annualized Return: {portfolio_details['annualized_return']:.2%}")
    print(f"Annualized Volatility: {portfolio_details['annualized_volatility']:.2%}")
    print(f"Sharpe Ratio: {portfolio_details['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {portfolio_details['sortino_ratio']:.2f}")
    print(f"Information Ratio: {portfolio_details['information_ratio']:.2f}")


def test_optimizer():
    tickers = ['SPY US', 'IVY US', 'VO US', '510050 CH']
    data_file = r'C:\Personal\Personal documents\Github\FolioLLM\data\test_prices.xlsx'
    risk_free_rate = 0.05  # 5% annual risk-free rate

    optimizer = PortfolioOptimizer(tickers, data_file, risk_free_rate)
    portfolio_details = optimizer.get_portfolio_details()

    print("Weights:")
    print(portfolio_details['weights_table'])
    print(f"Annualized Return: {portfolio_details['annualized_return']:.2%}")
    print(f"Annualized Volatility: {portfolio_details['annualized_volatility']:.2%}")
    print(f"Sharpe Ratio: {portfolio_details['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {portfolio_details['sortino_ratio']:.2f}")
    print(f"Information Ratio: {portfolio_details['information_ratio']:.2f}")


