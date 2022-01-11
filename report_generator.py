import os
from datetime import datetime, timedelta
import yahoo_fin.stock_info as si
from data_loader import DataLoader
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class ReportGenerator:
    def __init__(self, tickers, fmp_api_key):
        """
        Initiating method.
        :param tickers: list of strings, Contains tickers of stocks to analyze
        :param fmp_api_key: str, API Key for Financial modelling prep API
        """
        self.tickers = tickers
        self.fmp_api_key = fmp_api_key

        self.price_data = {}
        self.data = None
        self.reports = []
        self.ref_date = datetime.today().date().strftime('%d-%m-%Y')
        self.fig_size = (19.5, 26)

        for t in tickers:
            os.makedirs(f'Results/{t}', exist_ok=True)
            self.reports.append(PdfPages(f'Results/{t}/Stock Valuation of {t} - {self.ref_date}.pdf'))

    def run(self):
        # Load price data
        self.load_price_data()

        # Load fundamental data
        loader = DataLoader(self.tickers, self.fmp_api_key)
        self.data = loader.run()

        pass

    def load_price_data(self):
        for t in self.tickers:
            today = (datetime.today().date() + timedelta(days=1)).strftime('%m/%d/%Y')
            past = (datetime.today().date() - timedelta(days=5 * 365.25)).strftime('%m/%d/%Y')
            prices = si.get_data(t, start_date=past, end_date=today)['close'] \
                .reset_index() \
                .rename(columns={'index': 'Date', 'close': 'Price'})
            self.price_data[t] = prices

    def