from datetime import datetime, timedelta
import yahoo_fin.stock_info as si
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class ResultPlotter:
    def __init__(self, data):
        self.data = data
        self.tickers = data.index.values.tolist()
        self.ref_date = datetime.today().date().strftime('%m/%d/%Y')
        self.price_data = {}
        self.columns_to_display = ['EPS_TTM', 'EPS_Growth_by_Estimate', 'EPS_Growth_by_Equity', 'EPS_Growth',
                                   'Future_EPS', 'Future_PE_by_History', 'Future_PE_by_EPS_Growth', 'Future_PE',
                                   'Future_Price', 'Action_Price_Safe', 'Current_Price']

    def run(self):
        self.load_price_data()
        self.plot_results_by_company()

    def load_price_data(self):
        for t in self.tickers:
            today = (datetime.today().date() + timedelta(days=1)).strftime('%m/%d/%Y')
            past = (datetime.today().date() - timedelta(days=5 * 365.25)).strftime('%m/%d/%Y')
            prices = si.get_data(t, start_date=past, end_date=today)['close'] \
                .reset_index() \
                .rename(columns={'index': 'Date', 'close': 'Price'})
            self.price_data[t] = prices

    def plot_results_by_company(self):
        pdf = PdfPages('Results/New Money Stock Valuation.pdf')
        for t in self.tickers:
            fig, axes = plt.subplots(1, 2, figsize=(15, 10), squeeze=False, gridspec_kw={'width_ratios': [1, 3]})
            fig.subplots_adjust(top=0.9, bottom=0.1)
            fig.suptitle(f'Valuation analysis for {t} - {self.ref_date}', fontsize=24, y=0.98)

            table_data = self.data.loc[t, self.columns_to_display].to_frame('Value')
            table_data['Value'] = table_data['Value'].apply(lambda x: f'{x:.4}')
            table_data.index = [x.replace('_', ' ') for x in table_data.index]

            cell_text = []
            for row in range(len(table_data)):
                cell_text.append([table_data.index[row], table_data.iloc[row].values[0]])

            tab = axes[0, 0].table(cellText=cell_text, loc='center', colLabels=['Name', 'Value'])
            tab.auto_set_font_size(False)
            tab.set_fontsize(14)
            tab.auto_set_column_width(col=list(range(2)))
            tab.scale(1, 4)
            axes[0, 0].axis("off")

            axes[0, 1].plot(self.price_data[t]['Date'], self.price_data[t]['Price'], label=f'Price of {t}')
            axes[0, 1].plot(self.price_data[t]['Date'],
                            [self.data.loc[t, 'Action_Price_Safe']] * len(self.price_data[t]),
                            '--',
                            label=f'Safe Action Price of {t}')
            axes[0, 1].plot(self.price_data[t]['Date'],
                            [self.data.loc[t, 'Future_Price']] * len(self.price_data[t]),
                            '--',
                            label=f'Estimated Future Price of {t}')
            axes[0, 1].legend(fontsize=16)
            axes[0, 1].grid(True)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
        pdf.close()