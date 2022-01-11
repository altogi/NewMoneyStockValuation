import numpy as np


class DCFValuation:
    def __init__(self, data, span_years=10, desired_returns=0.15, safety_margin=0.5, selling_multiple=10):
        """
        Initiating method of the class
        :param data: pandas DataFrame with previously loaded data. See data_loader.py for its generation.
        :param span_years: int, Number of years in the future in which to carry out predictions.
        :param desired_returns: float, Desired returns, ranging from 0 to 1.0
        :param safety_margin: float, Safety margin to be applied to discount future cash flows
        :param selling_multiple: int, Estimates the selling price after span_years, when multiplied by the last cash flow.
        """
        try:
            print('DCFValuation: Initiating object.')
            required_cols = ['Free_Cash_Flow', 'Cash_Flow_Years', 'Current_Cash_and_Investments']
            self.data = None
            if not all([col in data.columns for col in required_cols]) or len(data) == 0 \
                    or len(data.dropna(subset=required_cols)) == 0:
                raise TypeError('Invalid or empty input dataframe.')
            self.data = data.dropna(subset=required_cols)
            self.span_years = span_years
            self.desired_returns = desired_returns
            self.safety_margin = safety_margin
            self.selling_multiple = selling_multiple

        except Exception as e:
            print(f'Error initiating DCFValuation. Stack trace\n: {e}')

    def run(self):
        if self.data is not None:
            self.calculate_future_cashflows()
            self.calculate_cashflow_of_sale()
            self.discount_cash_flows()
            self.sum_for_intrinsic_value()
            self.calculate_action_price()
        return self.data

    def calculate_future_cashflows(self):
        """
        Estimates future cash flows based on previous values, and propagates such growth into the future.
        """
        try:
            print('DCFValuation: Calculating Cashflow Growth Rate and Future Free Cashflows.')
            # Estimate Free Cashflow Growth Rate
            self.data['Cash_Flow_Growth'] = self.data[['Free_Cash_Flow', 'Cash_Flow_Years']].apply(
                lambda row: self.calculate_yearly_growth(row[0], row[1], method='mean'), axis=1)

            self.data['Starting_Cash_Flow'] = self.data[['Free_Cash_Flow']].apply(lambda x: x[0][0][0]).values[0]
            for yr in range(self.span_years):
                self.data[f'Cash_Flow_Year_{yr + 1}'] = self.data['Starting_Cash_Flow'] * (
                    (self.data['Cash_Flow_Growth'] + 1) ** (yr + 1))

        except Exception as e:
            print(f'Error calculating Cashflow Growth Rate with DCFValuation. Stack trace\n: {e}')

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
        values = [v for v in values[0] if v > 0]
        years = [y for i, y in enumerate(years[0]) if values[i] > 0]
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
        return growth

    def calculate_cashflow_of_sale(self):
        """
        Define a field Cash_Flow_Sale, calculated as a multiple self.selling_multiple of the last cash flow
        """
        try:
            print('DCFValuation: Calculating cashflow of sale as a multiple of the last cashflow.')
            self.data['Cash_Flow_Sale'] = self.selling_multiple * self.data[f'Cash_Flow_Year_{self.span_years}']

        except Exception as e:
            print(f'Error calculating cashflow of sale with DCFValuation. Stack trace\n: {e}')

    def discount_cash_flows(self):
        """
        Discount future cash flows (including sale) according to desired return rate
        """
        try:
            print('DCFValuation: Discounting all future cash flows.')
            for yr in range(self.span_years):
                discount_rate = (1 - self.desired_returns) ** (yr + 1)
                self.data[f'Discounted_Cash_Flow_Year_{yr + 1}'] = self.data[f'Cash_Flow_Year_{yr + 1}'] * discount_rate

            self.data['Discounted_Cash_Flow_Sale'] = self.data['Cash_Flow_Sale'] * (1 + self.desired_returns) ** (
                -self.span_years)
        except Exception as e:
            print(f'Error discounting all future cash flows with DCFValuation. Stack trace\n: {e}')

    def sum_for_intrinsic_value(self):
        """
        Sum all discounted future cash flows, as well as the company's current cash and investments, to obtain the
        intrinsic value of the company
        """
        try:
            print('DCFValuation: Adding all discounted future cash flows, and adding current assets.')
            columns_to_add = ['Discounted_Cash_Flow_Sale', 'Current_Cash_and_Investments'] + \
                             [f'Discounted_Cash_Flow_Year_{yr + 1}' for yr in range(self.span_years)]
            self.data['Intrinsic_Value'] = self.data[columns_to_add].sum(axis=1)

        except Exception as e:
            print(f'Error adding all discounted future cash flows with DCFValuation. Stack trace\n: {e}')

    def calculate_action_price(self):
        """
        Divide the intrinsic value by the number of issued shares, and apply the desired margin of safety, to determine
        the safe action price of the stock.
        """
        try:
            print('DCFValuation: Calculating Action Price based on intrinsic value, number of shares, and margin of safety.')
            self.data['Action_Price'] = self.data['Intrinsic_Value'] / self.data['Shares_Issued']
            self.data['Action_Price_Safe'] = self.data['Action_Price'] * (1 - self.safety_margin)
        except Exception as e:
            print(f'Error calculating action price with DCFValuation. Stack trace\n: {e}')