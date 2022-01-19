import numpy as np


class IntrinsicValueWithEPS:
    """
    Given a series of listed stocks, this class calculates a safe action price for each. With this price it is expected
    to obtain a desired return in a specified time span. It proceeds by estimating the growth rate of the earnings per
    share (EPS) of each stock, and applying this growth to the current EPS value, estimating the future EPS value. Then,
    it takes this estimated future EPS value and it multiplies it with an estimated P/E ratio, in order to estimate the
    future price. With this future price, the desired returns are propagated backwards, in order to determine the current
    price at which to buy to generate the desired returns. Finally a margin of safety is added to this action price.
    This method is extracted from: https://www.youtube.com/watch?v=QtRbk38hYMk&list=PLLtX9MQAlFKxoNZtGllOie9wUPdWNeVYT
    """
    def __init__(self, data, span_years=10, desired_returns=0.15, safety_margin=0.5):
        """
        Initiating method of the class
        :param data: pandas DataFrame with previously loaded data. See data_loader.py for its generation.
        :param span_years: int, Number of years in the future in which to carry out predictions.
        :param desired_returns: float, Desired returns, ranging from 0 to 1.0
        :param safety_margin: float, Safety margin to be applied to the backpropagated price.
        """
        try:
            print('ActionPriceCalculator: Initiating object.')
            required_cols = ['EPS_TTM', 'Past_Equities', 'Equity_Years', 'Growth_Estimate', 'PE_Ratio_High',
                             'PE_Ratio_Low', 'Current_Price']
            self.data = None
            if not all([col in data.columns for col in required_cols]) or len(data) == 0 \
                    or len(data.dropna(subset=required_cols)) == 0:
                raise TypeError('Invalid or empty input dataframe.')
            self.data = data.dropna(subset=required_cols)
            self.span_years = span_years
            self.desired_returns = desired_returns
            self.safety_margin = safety_margin

        except Exception as e:
            print(f'Error initiating ActionPriceCalculator. Stack trace\n: {e}')

    def run(self):
        """
        Main method of the class. Executes all the mentioned steps in order.
        :return: self.data, pandas DataFrame with results.
        """
        if self.data is not None:
            self.calculate_eps_growth()
            self.calculate_future_eps()
            self.calculate_future_price()
            self.calculate_action_price()
        return self.data

    def calculate_eps_growth(self):
        """
        Estimates EPS Growth rate with two methods. One method takes the growth of the company's equity in the last
        years. The other method estimated EPS growth with the company's growth as estimated by analysts. The minimum of
        these two growth rates is taken.
        """
        try:
            print('ActionPriceCalculator: Calculating EPS Growth Rate.')
            # Estimate EPS Growth as Equity Growth
            self.data['EPS_Growth_by_Equity'] = self.data[['Past_Equities', 'Equity_Years']].apply(
                lambda row: self.calculate_yearly_growth(row[0], row[1]), axis=1)

            # Estimate EPS Growth from Growth Estimates
            self.data['EPS_Growth_by_Estimate'] = self.data['Growth_Estimate']

            # Calculate EPS Growth as the minimum from both
            self.data['EPS_Growth'] = self.data[['EPS_Growth_by_Equity', 'EPS_Growth_by_Estimate']].apply(
                lambda row: [x is not None for x in row][0] if any([x is None for x in row]) else min(row), axis=1)

            # If EPS_Growth_by_Equity is negative but EPS_Growth_by_Equity is not, take the latter
            self.data['EPS_Growth'] = self.data[['EPS_Growth', 'EPS_Growth_by_Estimate']].apply(
                lambda row: row[1] if (row[0] < 0 and row[1] > 0) else row[0], axis=1)

        except Exception as e:
            print(f'Error calculating EPS Growth Rate with ActionPriceCalculator. Stack trace\n: {e}')

    @staticmethod
    def calculate_yearly_growth(values, years, only_furthest=False, method='mean'):
        """
        Auxiliar method that calculates the minimum yearly growth rate of a sequential series of values,
        split along a sequential series of years. If only_furthest is False, all yearly growth rates are taken by
        comparing with the last value of the series. Otherwise, only the oldest value is taken.
        :param values: 2D list, contains values from which to obtain minimum yearly growth.
        :param years: 2D list, contains year of each value
        :param only_furthest: bool, Determines whether to calculate only one rate (True) or all possible (False)
        :param method: str, Determines mode of extracting growth rate, if only_furthest=False. Can be min, max, or mean.
        :return: growth: float from 0 to 1.0, contains minimum yearly growth rate of the list.
        """
        values = values[0][0]
        years = years[0][0]
        if only_furthest:
            growth = (values[0] / values[-1]) ** (1 / (years[0] - years[-1])) - 1
        else:
            growth_rates = [(values[0] / eq) ** (1 / (years[0] - yr)) - 1 for eq, yr in
                               zip(values[1:], years[1:])]
            if method == 'mean':
                growth = np.mean(growth_rates)
            elif method == 'max':
                growth = max(growth_rates)
            else:
                growth = min(growth_rates)

        # if only_furthest:
        #     growth = np.exp(np.log(values[0] / values[-1]) / (years[0] - years[-1])) - 1
        # else:
        #     growth_rates = [np.exp(np.log(values[0] / eq) / (years[0] - yr)) - 1 for eq, yr in
        #                        zip(values[1:], years[1:])]
        #     growth = min(growth_rates)
        return growth

    def calculate_future_eps(self):
        """
        Estimates future EPS of each company with its current EPS and with the estimated EPS growth.
        """
        try:
            print('ActionPriceCalculator: Calculating Future EPS.')
            self.data['Future_EPS'] = self.data[['EPS_TTM', 'EPS_Growth']].apply(
                lambda row: row[0] * (1 + row[1]) ** self.span_years, axis=1)
        except Exception as e:
            print(f'Error calculating future EPS with ActionPriceCalculator. Stack trace\n: {e}')

    def calculate_future_price(self):
        """
        Estimates future stock price for each company by first estimating the future P/E ratio of the stock and
        multiplying it by the estimated future EPS of the stock. The future P/E ratio is estimated with two competing
        methods. The first method simply duplicates the obtained EPS growth rate (in percentage points), whereas the
        second method computes the average between the maximum and the minimum P/E values in the past 5 years. The
        future P/E ratio is the minimum between the results of both methods.
        """
        try:
            print('ActionPriceCalculator: Calculating Future Price based on estimated future EPS and future P/E Ratio.')

            # Estimate future P/E based on estimated Equity Growth
            self.data['Future_PE_by_EPS_Growth'] = 200 * self.data['EPS_Growth']

            # Estimate future P/E based on P/E history
            self.data['Future_PE_by_History'] = 0.5 * (self.data['PE_Ratio_High'] + self.data['PE_Ratio_Low'])
            self.data['Future_PE'] = self.data[['Future_PE_by_EPS_Growth', 'Future_PE_by_History']].apply(
                lambda row: [x is not None for x in row][0] if any([x is None for x in row]) else min(row), axis=1)

            # Compute Future Price
            self.data['Future_Price'] = self.data['Future_EPS'] * self.data['Future_PE']

            # Correct Future PE according to whether the growth expectations are positive and the future price is
            # smaller than the current price
            self.data['Future_PE'] = self.data[['Current_Price', 'Future_Price', 'Future_PE_by_History', 'Future_PE']].apply(
                lambda row: row[2] if row[0] > row[1] and row[2] > 0 else row[3], axis=1)
            self.data['Future_Price'] = self.data['Future_EPS'] * self.data['Future_PE']
        except Exception as e:
            print(f'Error calculating future price with ActionPriceCalculator. Stack trace\n: {e}')

    def calculate_action_price(self):
        """
        The current action price as well as the safe action price are obtained. The action price is calculated by
        backpropagating the desired returns from the estimated future price. The safe action price simply applies the
        specified safety margin to the former.
        """
        try:
            print('ActionPriceCalculator: Calculating Action Price based on estimated future price.')
            inverse_growth = (1 + self.desired_returns) ** self.span_years
            self.data['Action_Price'] = self.data['Future_Price'] / inverse_growth

            self.data['Action_Price_Safe'] = self.data['Action_Price'] * (1 - self.safety_margin)

        except Exception as e:
            print(f'Error calculating action price with ActionPriceCalculator. Stack trace\n: {e}')
