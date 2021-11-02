import pandas as pd
import yahoo_fin.stock_info as si
from datetime import datetime, timedelta
import time


class DataLoader:
    """
    This class takes in a list of tickers and extracts relevant fundamental data for the associated companies using the
    yahoo_fin API. This fundamental data is namely, for each stock:
        - The Trailing 12 month Earnings per Share of the stock
        - A list of equity values in the past
        - A list of years corresponding to the equity values in the past
        - A growth estimate for the company (from 0 to 1)
        - The maximum P/E ratio in the past 5 years
        - The minimum P/E ratio in the past 5 years
        - The current closing price
    """
    def __init__(self, tickers):
        """
        Initiating method.
        :param tickers: list of strings, Contains tickers of stocks to analyze
        """
        self.tickers = tickers
        self.data = pd.DataFrame(index=self.tickers, columns=['EPS_TTM', 'Past_Equities', 'Years',
                                                              'Growth_Estimate', 'PE_Ratio_High',
                                                              'PE_Ratio_Low', 'Current_Price'])
        # Columns in self.data
        # 'EPS_TTM': Trailing 12 month Earnings per Share
        # 'Past_Equities': Equity values in the past
        # 'Years': Years corresponding to the equity values in the past
        # 'Growth_Estimate': Growth estimate for the company (from 0 to 1)
        # 'PE_Ratio_High': Maximum P/E ratio in the past 5 years
        # 'PE_Ratio_Low': Minimum P/E ratio in the past 5 years
        # 'Current_Price': Current closing price

    def run(self):
        """
        Main method of the class. Executes all the mentioned steps in order.
        :return: self.data, pandas DataFrame with results.
        """
        self.get_eps_ttm()
        time.sleep(60)
        self.get_equity_values()
        time.sleep(60)
        self.get_growth_estimates()
        time.sleep(60)
        self.get_pe_ratios()
        time.sleep(60)
        return self.data

    def get_eps_ttm(self):
        """
        Use yahoo_fin to obtain the EPS (TTM) for each ticker
        """
        t = self.tickers[0]
        try:
            print('DataLoader: Loading EPS (TTM).')
            for t in self.tickers:
                quote_table = si.get_quote_table(t, dict_result=True)
                self.data.loc[t, 'EPS_TTM'] = quote_table['EPS (TTM)']
        except Exception as e:
            print(f'Error loading EPS (TTM) with DataLoader. Perhaps ticker "{t}" was not found. Stack trace\n: {e}')

    def get_equity_values(self):
        """
        Use yahoo_fin to obtain previous values for each company's total equity, and the years that they correspond to.
        """
        try:
            print('DataLoader: Loading past and current Equity Values.')
            for t in self.tickers:
                balance_sheet = si.get_balance_sheet(t)
                years = [y.year for y in balance_sheet.columns]
                equities = balance_sheet.loc['totalStockholderEquity', :].values.tolist()

                self.data.loc[t, 'Past_Equities'] = [equities]
                self.data.loc[t, 'Years'] = [years]

        except Exception as e:
            print(f'Error loading past and current Equity Values with DataLoader. Stack trace\n: {e}')

    def get_growth_estimates(self):
        """
        Scrape the yahoo finance website in order to extract the estimated per annum growth of each company in the
        following years.
        """
        try:
            print('DataLoader: Extracting growth estimates.')
            for t in self.tickers:
                analysis = si.get_analysts_info(t)
                growth = analysis['Growth Estimates'].loc[4, t]
                try:
                    growth = float(str(growth).replace('%', '')) / 100
                except ValueError:
                    print(f'Estimate of growth for {t} was not found. Imputing with NA')
                    growth = None
                self.data.loc[t, 'Growth_Estimate'] = growth
        except Exception as e:
            print(f'Error extracting growth estimates with DataLoader. Stack trace\n: {e}')

    def get_pe_ratios(self):
        """
        Use the yahoo_fin API to get a historical of each company's price and earnings per share (EPS) in order to obtain
        its historical P/E ratio. From there, extract the minimum and the maximum values. If no EPS data could be found,
        the current P/E ratio (TTM) is taken
        """
        try:
            print('DataLoader: Extracting high and low P/E Ratios.')
            for t in self.tickers:
                today = (datetime.today().date() + timedelta(days=1)).strftime('%m/%d/%Y')
                past = (datetime.today().date() - timedelta(days=5 * 365.25)).strftime('%m/%d/%Y')
                prices = si.get_data(t, start_date=past, end_date=today)['close'] \
                                .reset_index() \
                                .rename(columns={'index': 'startdatetime'})
                earnings = pd.DataFrame(si.get_earnings_history(t))[['startdatetime', 'epsactual']]

                earnings['startdatetime'] = pd.to_datetime(earnings['startdatetime']).dt.date
                prices['startdatetime'] = pd.to_datetime(prices['startdatetime']).dt.date

                pe_data = earnings.merge(prices, on='startdatetime', how='left').dropna()
                pe_data['P/E Ratio'] = pe_data['close'] / pe_data['epsactual']

                if len(pe_data) > 0:
                    self.data.loc[t, 'PE_Ratio_High'] = pe_data['P/E Ratio'].max()
                    self.data.loc[t, 'PE_Ratio_Low'] = pe_data['P/E Ratio'].min()
                else:
                    quote_table = si.get_quote_table(t, dict_result=True)
                    self.data.loc[t, 'PE_Ratio_High'] = quote_table['PE Ratio (TTM)']
                    self.data.loc[t, 'PE_Ratio_Low'] = quote_table['PE Ratio (TTM)']

                self.data.loc[t, 'Current_Price'] = prices['close'].dropna().tail(1).values[0]
                time.sleep(60)
        except Exception as e:
            print(f'Error extracting high and low P/E Ratios with DataLoader. Stack trace\n: {e}')