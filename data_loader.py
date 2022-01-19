import pandas as pd
import yahoo_fin.stock_info as si
from datetime import datetime, timedelta
import time
import requests


class DataLoader:
    """
    This class takes in a list of tickers and extracts relevant fundamental data for the associated companies using the
    yahoo_fin API.
    """
    def __init__(self, tickers, fmp_api_key, limit=20, avoid_api=True):
        """
        Initiating method.
        :param tickers: list of strings, Contains tickers of stocks to analyze
        :param fmp_api_key: str, API Key for Financial modelling prep API
        :param: limit: int, Maximum number of years back to query data
        :param: avoid_api: bool, If True, no API is called and all data is loaded from a predefined file
        """
        self.tickers = tickers
        self.fmp_api_key = fmp_api_key
        self.limit = limit
        self.avoid_api = avoid_api

        self.array_cols = ['Past_Equities', 'Equity_Years', 'PE_Ratios', 'PE_Ratio_Dates', 'Free_Cash_Flow',
                           'Cash_Flow_Years', 'Long_Term_Debt', 'LT_Debt_Years', 'Capital_Expenditure', 'Capex_Years',
                           'Inventory_Years', 'Inventory', 'Earnings_Years', 'Earnings', 'Outstanding_Shares',
                           'Outs_Shares_Years', 'Current_Cash_and_Investments', 'Dividend_Yields', 'Ratio_Years',
                           'Gross_Margins', 'Operating_Margins', 'Pretax_Margins', 'Net_Margins']
        self.required_cols = ['EPS_TTM', 'Past_Equities', 'Equity_Years', 'Growth_Estimate', 'PE_Ratios',
                              'PE_Ratio_Dates', 'PE_Ratio_High', 'PE_Ratio_Low', 'Current_Price', 'Free_Cash_Flow',
                              'Cash_Flow_Years', 'Current_Cash_and_Investments', 'Market_Cap', 'Shares_Issued',
                              'Dividend_Yields', 'Long_Term_Debt', 'LT_Debt_Years', 'Equity', 'Assets',
                              'Capital_Expenditure', 'Capex_Years', 'Inventory_Years', 'Inventory', 'Earnings',
                              'Earnings_Years', 'Outstanding_Shares', 'Outs_Shares_Years', 'Ratio_Years',
                              'Gross_Margins', 'Operating_Margins', 'Pretax_Margins', 'Net_Margins']
        self.data = pd.DataFrame(index=self.tickers, columns=self.required_cols)
        for col in self.array_cols:
            self.data[col] = self.data[col].astype(object)

        self.data_tmp = None

        # Columns in self.data
        # 'EPS_TTM': Trailing 12 month Earnings per Share
        # 'Past_Equities': Equity values in the past
        # 'Equity_Years': Years corresponding to the equity values in the past
        # 'Growth_Estimate': Growth estimate for the company (from 0 to 1)
        # 'PE_Ratios': Values of past P/E ratios in the past 5 years
        # 'PE_Ratio_Dates': Dates corresponding to the previous dates
        # 'PE_Ratio_High': Maximum P/E ratio in the past 5 years
        # 'PE_Ratio_Low': Minimum P/E ratio in the past 5 years
        # 'Current_Price': Current closing price
        # 'Free_Cash_Flow': Free cash flow values
        # 'Cash_Flow_Years': Years corresponding to the free cash flow values in the past
        # 'Current_Cash_and_Investments': Current cash and short term investments of the company
        # 'Market_Cap': Current market capitalization
        # 'Shares_Issued': Number of issued shares
        # 'Dividend_Yields': Evolution of dividend yield, in percentage of price
        # 'Ratio_Years': Years with which to associate all ratios
        # 'Long_Term_Debt': Previous values of long term debt
        # 'LT_Debt_Years': Years corresponding to the values of LT debt
        # 'Equity': Current value of total stockholder equity
        # 'Assets': Current value of total company assets
        # 'Capital_Expenditure': Values of previous capital expenditure
        # 'Capex_Years': Years corresponding to the values of capital expenditure
        # 'Inventory': Evolution of the company's inventory
        # 'Inventory_Years': Years corresponding to the above inventory values
        # 'Outstanding_Shares': Evolution of number of outstanding shares
        # 'Outs_Shares_Years': Years corresponding to the above outstanding share values
        # 'Gross_Margins', 'Operating_Margins', 'Pretax_Margins', 'Net_Margins': Evolution of profit margins

    def run(self):
        """
        Main method of the class. Executes all the mentioned steps in order.
        :return: self.data, pandas DataFrame with results.
        """
        self.attempt_file_load()
        right_tickers, right_cols = self.check_data_archive()

        if not self.avoid_api or (self.avoid_api and (not right_cols or not right_tickers)):
            self.get_quote_table()
            self.get_balance_sheet()
            self.get_growth_estimates()
            self.get_pe_ratios()
            self.get_free_cash_flow()
            self.get_income_statements()
            self.get_share_evolution()
            self.get_other_ratios()
            self.update_archive()
        else:
            self.data = self.data_tmp.copy()

        self.data.reset_index().to_csv('Common/data_archive.csv', index=False)
        return self.data

    def attempt_file_load(self):
        """
        See if there is a data archive in the expected directory. Load it if possible and apply all necessary formatting.
        """
        try:
            print("DataLoader: Attempting to load data from file.")
            self.data_tmp = pd.read_csv('Common/data_archive.csv', index_col=['index'])
            for col in self.array_cols:
                if col in self.data_tmp.columns:
                    self.data_tmp[col] = self.data_tmp[col].apply(lambda x: eval(x))
        except FileNotFoundError:
            print('DataLoader: File not found. Using API.')

    def check_data_archive(self):
        """
        See if the loaded data archive coincides in terms of tickers and required columns. Return two boolean flags
        indicating such matching.
        """
        right_tickers, right_cols = False, False
        if self.data_tmp is not None:
            # Check cols
            right_cols = all([col in self.data_tmp.columns for col in self.required_cols])

            # Check tickers
            right_tickers = all([t in self.data_tmp.index for t in self.tickers])

        return right_tickers, right_cols

    def get_quote_table(self):
        """
        Use yahoo_fin and FMP API to extract data from each company's quote table
        """
        t = self.tickers[0]
        cap_multipliers = {'T': 10**12, 'B': 10**9, 'M': 10**6}
        try:
            print("DataLoader: Extracting data from each company's quote table.")
            for t in self.tickers:
                fmp_api_url = f'https://financialmodelingprep.com/api/v3/quote/{t}' + \
                              f'?apikey={self.fmp_api_key}'
                response = requests.get(fmp_api_url)
                if response.status_code == 200:
                    quote_table = response.json()[0]
                    self.data.loc[t, 'EPS_TTM'] = quote_table['eps']
                    self.data.loc[t, 'Market_Cap'] = quote_table['marketCap']

                    self.data.loc[t, 'Current_Price'] = quote_table['previousClose']
                    self.data.loc[t, 'Shares_Issued'] = quote_table['sharesOutstanding']
                else:
                    quote_table = si.get_quote_table(t, dict_result=True)
                    self.data.loc[t, 'EPS_TTM'] = quote_table['EPS (TTM)']  # obtain the EPS (TTM) for each ticker
                    cap_str = quote_table['Market Cap']
                    if cap_str[-1] in cap_multipliers.keys():
                        cap = cap_str[-1]
                    else:
                        cap = 'M'
                    self.data.loc[t, 'Market_Cap'] = float(cap_str[:-1]) * cap_multipliers[cap]

                    self.data.loc[t, 'Current_Price'] = quote_table['Previous Close']
                    self.data.loc[t, 'Shares_Issued'] = self.data.loc[t, 'Market_Cap'] / self.data.loc[t, 'Current_Price']
                time.sleep(10)
        except Exception as e:
            print(f"Error extracting data from each company's quote table with DataLoader. "
                  f"Perhaps ticker '{t}' was not found. Stack trace\n: {e}")

    def get_balance_sheet(self):
        """
        Use FMP API to obtain previous values for each company's total equity, and the years that they correspond to.
        Also, extract the current cash and short term investments of the company.
        """
        try:
            print('DataLoader: Extracting data from balance sheet.')
            for t in self.tickers:
                fmp_api_url = f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{t}' + \
                              f'?limit={self.limit}&apikey={self.fmp_api_key}'
                response = requests.get(fmp_api_url)
                if response.status_code == 200:
                    balance_sheet = pd.DataFrame(response.json())

                    years = pd.to_datetime(balance_sheet['date']).dt.year.to_list()
                    equities = balance_sheet['totalStockholdersEquity'].to_list()
                    cashes = balance_sheet['cashAndShortTermInvestments'].to_list()
                    debts = balance_sheet['longTermDebt'].to_list()
                    inventories = balance_sheet['inventory'].to_list()
                    if years[-1] < years[0]:
                        years.reverse()
                        equities.reverse()
                        cashes.reverse()
                        debts.reverse()
                        inventories.reverse()
                    self.data.loc[t, 'Past_Equities'] = [[equities]]
                    self.data.loc[t, 'Equity_Years'] = [[years]]
                    self.data.loc[t, 'Current_Cash_and_Investments'] = [[cashes]]
                    self.data.loc[t, 'Long_Term_Debt'] = [[debts]]
                    self.data.loc[t, 'LT_Debt_Years'] = [[years]]
                    self.data.loc[t, 'Inventory'] = [[inventories]]
                    self.data.loc[t, 'Inventory_Years'] = [[years]]

                    self.data.loc[t, 'Equity'] = equities[0]
                    self.data.loc[t, 'Assets'] = balance_sheet['totalAssets'][0]
                    time.sleep(10)
                else:
                    print(f'Invalid FMP API response for {t}')

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
                time.sleep(10)
        except Exception as e:
            print(f'Error extracting growth estimates with DataLoader. Stack trace\n: {e}')

    def get_pe_ratios(self, method='minmax'):
        """
        Use the yahoo_fin API to get a historical of each company's price and earnings per share (EPS) in order to
        obtain its historical P/E ratio. From there, extract the minimum and the maximum values. If no EPS data could be
        found, the current P/E ratio (TTM) is taken
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
                fmp_api_url = f'https://financialmodelingprep.com/api/v3/key-metrics/{t}' + \
                              f'?limit={self.limit}&apikey={self.fmp_api_key}'
                response = requests.get(fmp_api_url)
                if response.status_code == 200:
                    pe_data = pd.DataFrame(response.json())
                    pe_data['P/E Ratio'] = pe_data['peRatio']
                    pe_data['startdatetime'] = pd.to_datetime(pe_data['date'])
                else:
                    today = (datetime.today().date() + timedelta(days=1)).strftime('%m/%d/%Y')
                    past = (datetime.today().date() - timedelta(days=5 * 365.25)).strftime('%m/%d/%Y')
                    prices = si.get_data(t, start_date=past, end_date=today)['close'] \
                        .reset_index() \
                        .rename(columns={'index': 'startdatetime'})

                    # Obtain EPS TTM
                    earnings = pd.DataFrame(si.get_earnings_history(t))[['startdatetime', 'epsactual']]
                    earnings['startdatetime'] = pd.to_datetime(earnings['startdatetime']).dt.date
                    earnings = earnings \
                        .drop_duplicates(subset='startdatetime') \
                        .sort_values(by='startdatetime', ascending=True)
                    earnings['eps_ttm'] = earnings['epsactual'].rolling(4).sum()

                    # Merge with prices
                    prices['startdatetime'] = pd.to_datetime(prices['startdatetime']).dt.date
                    pe_data = earnings.merge(prices, on='startdatetime', how='left').dropna()

                    pe_data['P/E Ratio'] = pe_data['close'] / pe_data['eps_ttm']

                if len(pe_data) > 0:
                    years = pe_data['startdatetime'].dt.year.to_list()
                    values = pe_data['P/E Ratio'].to_list()
                    if years[-1] < years[0]:
                        years.reverse()
                        values.reverse()
                    self.data.loc[t, 'PE_Ratio_Dates'] = [[years]]
                    self.data.loc[t, 'PE_Ratios'] = [[values]]

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

                time.sleep(10)
        except Exception as e:
            print(f'Error extracting high and low P/E Ratios with DataLoader. Stack trace\n: {e}')

    def get_free_cash_flow(self):
        """
        Use FMP and yahoo_fin to obtain previous values for each company's free cash flow, and the years that they
        correspond to.
        """
        try:
            print('DataLoader: Loading past and current Free Cash Flow Values.')
            for t in self.tickers:
                fmp_api_url = f'https://financialmodelingprep.com/api/v3/cash-flow-statement/{t}' + \
                              f'?limit={self.limit}&apikey={self.fmp_api_key}'
                response = requests.get(fmp_api_url)
                if response.status_code == 200:
                    cash_flows = pd.DataFrame(response.json())
                    years = pd.to_datetime(cash_flows['date']).dt.year.to_list()
                    values1 = cash_flows['freeCashFlow'].to_list()
                    values2 = cash_flows['capitalExpenditure'].to_list()
                    if years[-1] < years[0]:
                        years.reverse()
                        values1.reverse()
                        values2.reverse()
                    self.data.loc[t, 'Free_Cash_Flow'] = [[values1]]
                    self.data.loc[t, 'Cash_Flow_Years'] = [[years]]
                    self.data.loc[t, 'Capital_Expenditure'] = [[values2]]
                    self.data.loc[t, 'Capex_Years'] = [[years]]
                else:
                    cash_flows = si.get_cash_flow(t).T
                    cash_flows['Free_Cash_Flow'] = cash_flows['totalCashFromOperatingActivities'] + \
                        cash_flows['capitalExpenditures']
                    self.data.loc[t, 'Free_Cash_Flow'] = [[cash_flows['Free_Cash_Flow'].tolist()]]
                    self.data.loc[t, 'Cash_Flow_Years'] = [[[x.year for x in cash_flows.index.tolist()]]]
                    self.data.loc[t, 'Capital_Expenditure'] = [cash_flows['capitalExpenditure'].tolist()]
                    self.data.loc[t, 'Capex_Years'] = [[x.year for x in cash_flows.index.tolist()]]
                time.sleep(10)
        except Exception as e:
            print(f'Error loading past and current Free Cash Flow Values with DataLoader. Stack trace\n: {e}')

    def get_income_statements(self):
        try:
            print('DataLoader: Extracting data from income statement.')
            for t in self.tickers:
                fmp_api_url = f'https://financialmodelingprep.com/api/v3/income-statement/{t}' + \
                              f'?limit={self.limit}&apikey={self.fmp_api_key}'
                response = requests.get(fmp_api_url)
                if response.status_code == 200:
                    income_statement = pd.DataFrame(response.json())
                    years = pd.to_datetime(income_statement['date']).dt.year.to_list()
                    values = income_statement['netIncome'].to_list()
                    if years[-1] < years[0]:
                        years.reverse()
                        values.reverse()
                    self.data.loc[t, 'Earnings'] = [[values]]
                    self.data.loc[t, 'Earnings_Years'] = [[years]]
                else:
                    print(f'Invalid FMP API response for {t}')

        except Exception as e:
            print(f'Error extracting data from income statement with DataLoader. Stack trace\n: {e}')

    def get_share_evolution(self):
        try:
            print('DataLoader: Extracting outstanding share data.')
            for t in self.tickers:
                fmp_api_url = f'https://financialmodelingprep.com/api/v3/enterprise-values/{t}' + \
                              f'?limit={self.limit}&apikey={self.fmp_api_key}'
                response = requests.get(fmp_api_url)
                if response.status_code == 200:
                    enterprise_values = pd.DataFrame(response.json())
                    years = pd.to_datetime(enterprise_values['date']).dt.year.to_list()
                    values = enterprise_values['numberOfShares'].to_list()
                    if years[-1] < years[0]:
                        years.reverse()
                        values.reverse()
                    self.data.loc[t, 'Outstanding_Shares'] = [[values]]
                    self.data.loc[t, 'Outs_Shares_Years'] = [[years]]
                else:
                    print(f'Invalid FMP API response for {t}')
        except Exception as e:
            print(f'Error extracting outstanding share data with DataLoader. Stack trace\n: {e}')

    def get_other_ratios(self):
        try:
            print('DataLoader: Extracting company ratio data.')
            for t in self.tickers:
                # Separate query for dividend yield
                fmp_api_url = f'https://financialmodelingprep.com/api/v3/ratios/{t}' + \
                              f'?limit={self.limit}&apikey={self.fmp_api_key}'
                response = requests.get(fmp_api_url)
                if response.status_code == 200:
                    ratio_table = pd.DataFrame(response.json())
                    dividend = ratio_table['dividendYield'].to_list()
                    gross = ratio_table['grossProfitMargin'].to_list()
                    operating = ratio_table['operatingProfitMargin'].to_list()
                    pretax = ratio_table['pretaxProfitMargin'].to_list()
                    net = ratio_table['netProfitMargin'].to_list()
                    years = pd.to_datetime(ratio_table['date']).dt.year.to_list()
                    if years[-1] < years[0]:
                        years.reverse()
                        dividend.reverse()
                        gross.reverse()
                        operating.reverse()
                        pretax.reverse()
                        net.reverse()
                    self.data.loc[t, 'Dividend_Yields'] = [[dividend]]
                    self.data.loc[t, 'Gross_Margins'] = [[gross]]
                    self.data.loc[t, 'Operating_Margins'] = [[operating]]
                    self.data.loc[t, 'Pretax_Margins'] = [[pretax]]
                    self.data.loc[t, 'Net_Margins'] = [[net]]
                    self.data.loc[t, 'Ratio_Years'] = [[years]]
                else:
                    print(f'Invalid FMP API response for {t}')
        except Exception as e:
            print(f'Error extracting company ratio data with DataLoader. Stack trace\n: {e}')

    def update_archive(self):
        """
        Update the data archive to include any new data obtained with the APIs
        """
        try:
            print('DataLoader: Updating data to archive.')
            if self.data_tmp is not None:
                for t in self.data_tmp.index:
                    if t not in self.tickers:
                        self.data.loc[t, self.required_cols] = self.data_tmp.loc[t, self.required_cols]
                    else:
                        for col in self.array_cols:
                            if col in self.data_tmp.columns:
                                archive_list = self.data_tmp.loc[t, col][0][0]
                                newest_list = self.data.loc[t, col][0][0]
                                updated_list = self.compare_and_unite(archive_list, newest_list)
                                self.data.loc[t, col] = [[updated_list]]
        except Exception as e:
            print(f'Error updating data to archive with DataLoader. Stack trace\n: {e}')

    @staticmethod
    def compare_and_unite(l_old, l_new):
        """
        Compares two lists
        :param l_old:
        :param l_new:
        :return:
        """
        if l_new[0] in l_old:
            cut = l_old.index(l_new[0])
            if cut > 0:
                l_united = l_old[:cut] + l_new
            else:
                l_united = l_new
        else:
            l_united = l_old + l_new
        return l_united
