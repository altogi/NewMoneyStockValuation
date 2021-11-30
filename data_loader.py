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
        self.data = pd.DataFrame(index=self.tickers, columns=['EPS_TTM', 'Past_Equities', 'Equity_Years',
                                                              'Growth_Estimate', 'PE_Ratio_High', 'PE_Ratio_Low',
                                                              'Current_Price', 'Free_Cash_Flow', 'Cash_Flow_Years',
                                                              'Current_Cash_and_Investments', 'Market_Cap'])
        self.data['Past_Equities'] = self.data['Past_Equities'].astype(object)
        self.data['Equity_Years'] = self.data['Equity_Years'].astype(object)
        self.data['Free_Cash_Flow'] = self.data['Free_Cash_Flow'].astype(object)
        self.data['Cash_Flow_Years'] = self.data['Cash_Flow_Years'].astype(object)

        # Columns in self.data
        # 'EPS_TTM': Trailing 12 month Earnings per Share
        # 'Past_Equities': Equity values in the past
        # 'Equity_Years': Years corresponding to the equity values in the past
        # 'Growth_Estimate': Growth estimate for the company (from 0 to 1)
        # 'PE_Ratio_High': Maximum P/E ratio in the past 5 years
        # 'PE_Ratio_Low': Minimum P/E ratio in the past 5 years
        # 'Current_Price': Current closing price
        # 'Free_Cash_Flow': Free cash flow values
        # 'Cash_Flow_Years': Years corresponding to the free cash flow values in the past
        # 'Current_Cash_and_Investments': Current cash and short term investments of the company
        # 'Market_Cap': Current market capitalization
        # 'Shares_Issued': Number of issued shares

    def run(self):
        """
        Main method of the class. Executes all the mentioned steps in order.
        :return: self.data, pandas DataFrame with results.
        """
        self.get_quote_table()
        time.sleep(60)
        self.get_balance_sheet()
        time.sleep(60)
        self.get_growth_estimates()
        time.sleep(60)
        self.get_pe_ratios()
        time.sleep(60)
        self.get_free_cash_flow()
        return self.data

    def get_quote_table(self):
        """
        Use yahoo_fin to extract data from each company's quote table
        """
        t = self.tickers[0]
        cap_multipliers = {'T': 10**12, 'B': 10**9, 'M': 10**6}
        try:
            print("DataLoader: Extracting data from each company's quote table.")
            for t in self.tickers:
                quote_table = si.get_quote_table(t, dict_result=True)
                self.data.loc[t, 'EPS_TTM'] = quote_table['EPS (TTM)'] # obtain the EPS (TTM) for each ticker
                cap_str = quote_table['Market Cap']
                self.data.loc[t, 'Market_Cap'] = float(cap_str[:-1]) * cap_multipliers[cap_str[-1]]
                time.sleep(20)
        except Exception as e:
            print(f"Error extracting data from each company's quote table with DataLoader. Perhaps ticker '{t}' was not found. Stack trace\n: {e}")

    def get_balance_sheet(self):
        """
        Use yahoo_fin to obtain previous values for each company's total equity, and the years that they correspond to.
        Also, extract the current cash and short term investments of the company.
        """
        try:
            print('DataLoader: Extracting data from balance sheet.')
            for t in self.tickers:
                balance_sheet = si.get_balance_sheet(t)
                years = [y.year for y in balance_sheet.columns]
                equities = balance_sheet.loc['totalStockholderEquity', :].values.tolist()

                self.data.loc[t, 'Past_Equities'] = [equities]
                self.data.loc[t, 'Equity_Years'] = [years]

                self.data.loc[t, 'Current_Cash_and_Investments'] = \
                    balance_sheet.loc['shortTermInvestments', balance_sheet.columns[0]] + \
                    balance_sheet.loc['cash', balance_sheet.columns[0]]
                time.sleep(20)

        except Exception as e:
            print(f'Error extracting data from balance sheet with DataLoader. Stack trace\n: {e}')

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
                time.sleep(20)
        except Exception as e:
            print(f'Error extracting growth estimates with DataLoader. Stack trace\n: {e}')

    def get_pe_ratios(self, method='q25'):
        """
        Use the yahoo_fin API to get a historical of each company's price and earnings per share (EPS) in order to obtain
        its historical P/E ratio. From there, extract the minimum and the maximum values. If no EPS data could be found,
        the current P/E ratio (TTM) is taken
        :param method: str, Determines what method to use in order to select P/E ratios from history. Can be 'minmax',
            'mean', 'median', 'q{x}' (for the xth quantile).
        """
        try:
            print('DataLoader: Extracting high and low P/E Ratios.')
            if method not in ['minmax', 'mean', 'median']:
                if method[0] == 'q':
                    if method[1:].isdigit() and int(method[1:]) < 100:
                        pass
                    else:
                        method = 'minmax'
                else:
                    method = 'minmax'

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
                    if method == 'minmax':
                        self.data.loc[t, 'PE_Ratio_High'] = pe_data['P/E Ratio'].max()
                        self.data.loc[t, 'PE_Ratio_Low'] = pe_data['P/E Ratio'].min()
                    elif method == 'mean':
                        self.data.loc[t, 'PE_Ratio_High'] = pe_data['P/E Ratio'].mean()
                        self.data.loc[t, 'PE_Ratio_Low'] = pe_data['P/E Ratio'].mean()
                    elif method == 'median':
                        self.data.loc[t, 'PE_Ratio_High'] = pe_data['P/E Ratio'].median()
                        self.data.loc[t, 'PE_Ratio_Low'] = pe_data['P/E Ratio'].median()
                    elif method[0] == 'q':
                        quant = int(method[1:]) / 100
                        self.data.loc[t, 'PE_Ratio_High'] = pe_data['P/E Ratio'].quantile(quant)
                        self.data.loc[t, 'PE_Ratio_Low'] = pe_data['P/E Ratio'].quantile(quant)
                else:
                    quote_table = si.get_quote_table(t, dict_result=True)
                    self.data.loc[t, 'PE_Ratio_High'] = quote_table['PE Ratio (TTM)']
                    self.data.loc[t, 'PE_Ratio_Low'] = quote_table['PE Ratio (TTM)']

                self.data.loc[t, 'Current_Price'] = prices['close'].dropna().tail(1).values[0]
                self.data.loc[t, 'Shares_Issued'] = self.data.loc[t, 'Market_Cap'] / self.data.loc[t, 'Current_Price']
                time.sleep(60)
        except Exception as e:
            print(f'Error extracting high and low P/E Ratios with DataLoader. Stack trace\n: {e}')

    def get_free_cash_flow(self):
        """
        Use yahoo_fin to obtain previous values for each company's free cash flow, and the years that they correspond to.
        """
        try:
            print('DataLoader: Loading past and current Free Cash Flow Values.')
            for t in self.tickers:
                cash_flows = si.get_cash_flow(t).T
                cash_flows['Free_Cash_Flow'] = cash_flows['totalCashFromOperatingActivities'] + cash_flows['capitalExpenditures']

                self.data.loc[t, 'Free_Cash_Flow'] = [[cash_flows['Free_Cash_Flow'].tolist()]]
                self.data.loc[t, 'Cash_Flow_Years'] = [[[x.year for x in cash_flows.index.tolist()]]]
                time.sleep(20)
        except Exception as e:
            print(f'Error loading past and current Free Cash Flow Values with DataLoader. Stack trace\n: {e}')

