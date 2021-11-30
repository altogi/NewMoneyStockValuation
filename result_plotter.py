from datetime import datetime, timedelta
import yahoo_fin.stock_info as si
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class ResultPlotter:
    def __init__(self, data, tickers):
        self.data = data
        self.tickers = tickers
        self.ref_date = datetime.today().date().strftime('%d-%m-%Y')
        self.price_data = {}

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
        for t in self.tickers:
            pdf = PdfPages(f'Results/Stock Valuation of {t} - {self.ref_date}.pdf')
            for i in range(len(self.data)):
                analysis = self.data[i]
                columns_to_display = analysis['Table_Cols']
                data = analysis['Data']
                price_levels = analysis['Price_Levels']
                price_labels = analysis['Price_Level_Names']
                fig_title = analysis['Title']

                fig, axes = plt.subplots(1, 2, figsize=(15, 10), squeeze=False, gridspec_kw={'width_ratios': [1, 3]})
                fig.subplots_adjust(top=0.9, bottom=0.1)
                fig.suptitle(fig_title + f' - {t}', fontsize=24, y=0.98)

                table_data = data.loc[t, columns_to_display].to_frame('Value')
                self.print_data_table(table_data, axes[0, 0])
                self.plot_price_lines(data, price_levels, price_labels, t, axes[0, 1])

                plt.tight_layout()
                pdf.savefig()
                plt.close()
            pdf.close()

    @staticmethod
    def print_data_table(table_data, ax):
        table_data['Value'] = table_data['Value'].apply(lambda x: f'{x:.4e}')
        table_data.index = [x.replace('_', ' ') for x in table_data.index]

        cell_text = []
        for row in range(len(table_data)):
            cell_text.append([table_data.index[row], table_data.iloc[row].values[0]])

        tab = ax.table(cellText=cell_text, loc='center', colLabels=['Name', 'Value'])
        tab.auto_set_font_size(False)
        tab.set_fontsize(14)
        tab.auto_set_column_width(col=list(range(2)))
        tab.scale(1, 4)
        ax.axis("off")

    def plot_price_lines(self, data, levels, labels, t, ax):
        ax.plot(self.price_data[t]['Date'], self.price_data[t]['Price'], label=f'Price of {t}')
        for lev, lab in zip(levels, labels):
            ax.plot(self.price_data[t]['Date'], [data.loc[t, lev]] * len(self.price_data[t]), '--', label=lab + f'{t}')
        ax.legend(fontsize=16)
        ax.grid(True)
