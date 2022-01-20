import os
import numpy as np
import pandas as pd
import config as conf
import Common.questions as qs
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yahoo_fin.stock_info as si
from data_loader import DataLoader
from eps_propagation_valuation import IntrinsicValueWithEPS
from discounted_cash_flow_valuation import DCFValuation
from fpdf import FPDF
from PyPDF2 import PdfFileMerger


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
        self.sp500_pe = None
        self.data = None
        self.reports = []
        self.ref_date = datetime.today().date().strftime('%d-%m-%Y')

        for t in tickers:
            os.makedirs(f'Results/{t}/Images', exist_ok=True)
            self.reports.append(FPDF(unit='in', format=(19.5, 26)))

    def run(self):
        # Load price data
        self.load_price_data()

        # Load fundamental data
        loader = DataLoader(self.tickers, self.fmp_api_key, avoid_api=True)
        self.data = loader.run()

        self.generate_checklist_page()

        self.generate_earnings_page()

        self.generate_pe_page()

        self.generate_cash_page()

        self.generate_debt_page()

        self.generate_cash_flows_page()

        self.generate_dividend_page()

        self.generate_book_value_page()

        self.generate_inventory_page()

        self.generate_sales_page()

        self.generate_profit_margins_page()

        self.generate_eps_propagation_page()

        self.generate_discounted_cashflow_page()

        self.generate_growth_phase_page()

        self.generate_2_minute_drill_page()

        self.save_and_appendix()

    def load_price_data(self):
        self.sp500_pe = pd.read_csv('Common/sp500_peratio.csv', parse_dates=['Date'], sep='\t')
        for t in self.tickers:
            today = (datetime.today().date() + timedelta(days=1)).strftime('%m/%d/%Y')
            past = (datetime.today().date() - timedelta(days=5 * 365.25)).strftime('%m/%d/%Y')
            prices = si.get_data(t, start_date=past, end_date=today)['close'] \
                .reset_index() \
                .rename(columns={'index': 'Date', 'close': 'Price'})
            self.price_data[t] = prices

    def generate_checklist_page(self):
        for i, t in enumerate(self.tickers):
            pdf = self.reports[i]
            pdf.add_page()
            pdf.set_font('Courier', style='B', size=50)
            pdf.cell(18, 2, ln=2, align='C', txt=f'I. Sum of Good and Bad Attributes - {t}')
            pdf.ln(0.5)
            pdf.image(f'Common/checklist.jpg', w=19.15, x=0.5)

    def generate_earnings_page(self):
        for i, t in enumerate(self.tickers):
            # Generate graph 1 and save as image
            fig, ax1 = plt.subplots(1, 1, figsize=(15, 10))
            fig.suptitle(f'Earnings Growth Analysis - {t}', fontsize=24)

            ax1.set_xlabel('Years', fontsize=20)
            ax1.set_ylabel('Earnings', fontsize=20)
            x = self.data.loc[t, 'Earnings_Years'][0][0]
            y = self.data.loc[t, 'Earnings'][0][0]
            ax1.tick_params(axis='x', labelsize=18)
            ax1.tick_params(axis='y', labelsize=18)
            ax1.plot(x, y, linewidth=5)

            ax2 = ax1.twinx()
            ax2.set_ylabel('Earning Growth Rate [%]', fontsize=24)
            x = self.data.loc[t, 'Earnings_Years'][0][0]
            x = [0.5 * (x0 + x1) for x0, x1 in zip(x[:-1], x[1:])]
            y = self.data.loc[t, 'Earnings'][0][0]
            y = [((e1 / e0) - 1) * 100 for e0, e1 in zip(y[:-1], y[1:])]
            ax2.tick_params(axis='y', labelsize=18)
            ax2.bar(x, y, width=0.3, alpha=0.6)

            x = self.data.loc[t, 'Earnings_Years'][0][0]
            plt.xticks(x, x)
            plt.axhline(y=0, color="black", linestyle=":")
            plt.tight_layout()
            plt.savefig(f'Results/{t}/Images/earnings_growth.jpg')
            plt.close()

            # Generate graph 2 and save as image
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            fig.suptitle(f'Earnings versus Price - {t}', fontsize=24)
            x1 = self.data.loc[t, 'Earnings_Years'][0][0]
            x1 = [datetime(yr, 12, 31) for yr in x1]
            y1 = self.data.loc[t, 'Earnings'][0][0]
            y1 = [y / y1[0] for y in y1]
            ax.plot(x1, y1, ':', label='Earnings Growth', linewidth=5)

            x2 = self.price_data[t]['Date'].values
            y2 = self.price_data[t]['Price'].values
            y2 = [y / y2[0] for y in y2]
            ax.plot(x2, y2, label='Price Growth', linewidth=5)
            ax.set_xlabel('Date', fontsize=20)
            ax.legend(fontsize=20)

            ax.tick_params(axis='y', labelsize=18)
            ax.tick_params(axis='x', labelsize=18)

            plt.tight_layout()
            plt.savefig(f'Results/{t}/Images/earnings_vs_price.jpg')
            plt.close()

            # Create a new report page
            pdf = self.reports[i]
            pdf.add_page()
            pdf.set_font('Courier', style='B', size=50)
            pdf.cell(18, 2, ln=2, align='C', txt=f'II. Earnings Growth Analysis - {t}')
            pdf.ln(0.5)
            pdf.image(f'Results/{t}/Images/earnings_growth.jpg', h=9, x=2.5)
            pdf.ln(0.5)
            pdf.image(f'Results/{t}/Images/earnings_vs_price.jpg', h=9, x=2.5)

    def generate_pe_page(self):
        for i, t in enumerate(self.tickers):
            # Generate graph and save as image
            fig = plt.figure(figsize=(15, 10))
            fig.suptitle(f'Historical P/E Analysis - {t}', fontsize=24)

            gs = fig.add_gridspec(1, 2, width_ratios=(8, 2), left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05)
            ax = fig.add_subplot(gs[0, 0])
            ax_box = fig.add_subplot(gs[0, 1], sharey=ax)
            ax_box.tick_params(axis="y", labelleft=False)

            x = self.data.loc[t, 'PE_Ratio_Dates'][0][0]
            x = [datetime(yr, 12, 31) for yr in x]
            y1 = self.data.loc[t, 'PE_Ratios'][0][0]
            ax.plot(x, y1, label=f'P/E Ratio of {t}', linewidth=5)

            mask = self.sp500_pe['Date'] > min(x)
            x = self.sp500_pe.loc[mask, 'Date']
            y2 = self.sp500_pe.loc[mask, 'PE']
            ax.plot(x, y2, '-.', label=f'P/E Ratio of S&P500', linewidth=5)

            ax.tick_params(axis='y', labelsize=18)
            ax.tick_params(axis='x', labelsize=18)
            ax.set_xlabel('Date', fontsize=20)
            ax.set_ylabel('P/E Ratio', fontsize=20)
            ax.legend(fontsize=20)

            ax_box.boxplot(y1, vert=True, meanline=True)

            plt.savefig(f'Results/{t}/Images/historical_pe.jpg')
            plt.close()

            # Create a new report page
            pdf = self.reports[i]
            pdf.add_page()
            pdf.set_font('Courier', style='B', size=50)
            pdf.cell(18, 2, ln=2, align='C', txt=f'III. Historical P/E Analysis - {t}')
            pdf.ln(0.5)
            pdf.image(f'Results/{t}/Images/historical_pe.jpg', w=16, x=1.5)
            pdf.ln(0.5)

            # Collect data to print on page
            y = self.data.loc[t, 'Earnings'][0][0]
            y = [((e1 / e0) - 1) * 100 for e0, e1 in zip(y[:-1], y[1:])]
            dividend = self.data.loc[t, 'Dividend_Yields'][0][0][-1] * 100 if self.data.loc[t, 'Dividend_Yields'][0][0][
                                                                                  -1] is not None else None
            data = pd.DataFrame(data=
                                {'Name':
                                     ['P/E Ratio [-]',
                                      'Latest Earnings Growth [%]',
                                      'Average Earnings Growth [%]',
                                      'Dividend Yield [%]'],
                                 'Value':
                                     [self.price_data[t]['Price'].values[-1] / self.data.loc[t, 'EPS_TTM'],
                                      y[-1],
                                      np.mean(y),
                                      dividend]})

            # Print data on page
            self.output_df_to_pdf(pdf, data)

    def generate_cash_page(self):
        for i, t in enumerate(self.tickers):
            # Generate graph and save as image
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            fig.suptitle(f'Cash + Securities versus LT Debt - {t}', fontsize=24)
            x1 = self.data.loc[t, 'Cash_Flow_Years'][0][0]
            y1 = self.data.loc[t, 'Current_Cash_and_Investments'][0][0]
            y1 = [y / y1[0] for y in y1]
            ax.plot(x1, y1, ':', label='Cash Growth', linewidth=5)

            x2 = self.data.loc[t, 'LT_Debt_Years'][0][0]
            y2 = self.data.loc[t, 'Long_Term_Debt'][0][0]
            y2 = [y / y2[0] for y in y2]
            ax.plot(x2, y2, label='LT Debt Growth', linewidth=5)
            ax.set_xlabel('Year', fontsize=20)
            ax.legend(fontsize=20)

            ax.tick_params(axis='y', labelsize=18)
            ax.tick_params(axis='x', labelsize=18)

            plt.xticks(x1, x1)
            plt.tight_layout()
            plt.savefig(f'Results/{t}/Images/cash_vs_debt.jpg')
            plt.close()

            # Second figure
            fig, ax1 = plt.subplots(1, 1, figsize=(15, 10))
            fig.suptitle(f'Evolution of Net Cash and Outstanding Shares - {t}', fontsize=24)

            ax1.set_xlabel('Years', fontsize=20)
            ax1.set_ylabel('Net Cash', fontsize=20)
            x = self.data.loc[t, 'Cash_Flow_Years'][0][0]
            y1 = self.data.loc[t, 'Current_Cash_and_Investments'][0][0]
            y2 = self.data.loc[t, 'Long_Term_Debt'][0][0]
            y = [v1 - v2 for v1, v2 in zip(y1, y2)]
            ax1.tick_params(axis='x', labelsize=18)
            ax1.tick_params(axis='y', labelsize=18)
            ax1.plot(x, y, label='Net Cash', linewidth=5)

            ax2 = ax1.twinx()
            ax2.set_ylabel('Outstanding Shares', fontsize=24)
            x = self.data.loc[t, 'Outs_Shares_Years'][0][0]
            y = self.data.loc[t, 'Outstanding_Shares'][0][0]
            ax2.tick_params(axis='y', labelsize=18)
            ax2.plot(x, y, '--', label='Outstanding Shares', linewidth=5)
            ax2.legend(fontsize=20)

            plt.tight_layout()
            plt.xticks(x, x)
            plt.savefig(f'Results/{t}/Images/net_cash_vs_outstanding.jpg')
            plt.close()

            # Create a new report page
            pdf = self.reports[i]
            pdf.add_page()
            pdf.set_font('Courier', style='B', size=50)
            pdf.cell(18, 2, ln=2, align='C', txt=f'IV. Cash Position Analysis - {t}')
            pdf.ln(0.5)
            pdf.image(f'Results/{t}/Images/cash_vs_debt.jpg', w=16, x=2)
            pdf.ln(1)

            # Collect data to print on page
            net_cash = self.data.loc[t, 'Current_Cash_and_Investments'][0][0][-1] - self.data.loc[t, 'Long_Term_Debt'][0][0][-1]
            discounted_price = self.price_data[t]['Price'].values[-1] - (net_cash / self.data.loc[t, 'Shares_Issued'])
            discounted_pe = discounted_price / self.data.loc[t, 'EPS_TTM']
            data = pd.DataFrame(data=
                                {'Name': ['Current Price [$]', 'Net Cash-Discounted Price [$]', 'Current PE Ratio [-]',
                                          'Net Cash-Discounted P/E Ratio [-]'],
                                 'Value': [self.price_data[t]['Price'].values[-1], discounted_price,
                                           self.price_data[t]['Price'].values[-1] / self.data.loc[t, 'EPS_TTM'],
                                           discounted_pe]})

            # Print data on page
            self.output_df_to_pdf(pdf, data)

            pdf.add_page()
            pdf.set_font('Courier', style='B', size=50)
            pdf.cell(18, 2, ln=2, align='C', txt=f'IV. (CONT.) - {t}')
            pdf.ln(0.5)
            pdf.image(f'Results/{t}/Images/net_cash_vs_outstanding.jpg', w=16, x=2)

    def generate_debt_page(self):
        for i, t in enumerate(self.tickers):
            # Generate graph and save as image
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            fig.suptitle(f'LT Debt / Equity - {t}', fontsize=24)
            x1 = self.data.loc[t, 'Equity_Years'][0][0]
            y1 = self.data.loc[t, 'Past_Equities'][0][0]
            y2 = self.data.loc[t, 'Long_Term_Debt'][0][0]
            y = [v2 / v1 for v1, v2 in zip(y1, y2)]
            ax.plot(x1, y, linewidth=5)

            ax.tick_params(axis='y', labelsize=18)
            ax.tick_params(axis='x', labelsize=18)
            plt.tight_layout()
            plt.xticks(x1, x1)
            plt.savefig(f'Results/{t}/Images/debt_factor.jpg')
            plt.close()

            # Create a new report page
            pdf = self.reports[i]
            pdf.add_page()
            pdf.set_font('Courier', style='B', size=50)
            pdf.cell(18, 2, ln=2, align='C', txt=f'V. Debt Factor Analysis - {t}')
            pdf.ln(0.5)
            pdf.image(f'Results/{t}/Images/debt_factor.jpg', w=14, x=2.5)
            pdf.ln(0.5)

            # Collect data to print on page
            data = pd.DataFrame(data=
                                {'Name': ['Equity / Assets', 'Assets ='],
                                 'Value': [self.data.loc[t, 'Equity'] / self.data.loc[t, 'Assets'],
                                           'Equity (75%) + Liabilities (25%)']})

            # Print data on page
            self.output_df_to_pdf(pdf, data)

            # Question box
            self.question_box(pdf, 8, 'Is the debt due on demand? What type of debt is it?')

    def generate_cash_flows_page(self):
        for i, t in enumerate(self.tickers):
            # Generate graph and save as image
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            fig.suptitle(f'Cash Flows Evolution - {t}', fontsize=24)
            x1 = self.data.loc[t, 'Cash_Flow_Years'][0][0]
            y1 = self.data.loc[t, 'Free_Cash_Flow'][0][0]
            ax.plot(x1, y1, ':', label='Free Cash Flow', linewidth=5)

            x2 = self.data.loc[t, 'Capex_Years'][0][0]
            y2 = self.data.loc[t, 'Capital_Expenditure'][0][0]
            y2 = [-v2 for v2 in y2]
            ax.plot(x2, y2, label='Capital Expenditure', linewidth=5)
            ax.set_xlabel('Year', fontsize=20)
            ax.legend(fontsize=20)

            ax.tick_params(axis='y', labelsize=18)
            ax.tick_params(axis='x', labelsize=18)

            plt.xticks(x1, x1)
            plt.tight_layout()
            plt.savefig(f'Results/{t}/Images/cash_flows.jpg')
            plt.close()

            # Create a new report page
            pdf = self.reports[i]
            pdf.add_page()
            pdf.set_font('Courier', style='B', size=50)
            pdf.cell(18, 2, ln=2, align='C', txt=f'VI. Cash Flows Analysis - {t}')
            pdf.ln(0.5)
            pdf.image(f'Results/{t}/Images/cash_flows.jpg', w=15, x=2)

            # Question Box 1
            self.question_box(pdf, 9, f'What are the depreciation allowances?')

    def generate_dividend_page(self):
        for i, t in enumerate(self.tickers):
            # Generate graph 1 and save as image
            fig, ax1 = plt.subplots(1, 1, figsize=(15, 10))
            fig.suptitle(f'Dividend Analysis - {t}', fontsize=24)

            ax1.set_xlabel('Years', fontsize=20)
            ax1.set_ylabel('Dividend Yield [%]', fontsize=20)
            x = self.data.loc[t, 'Ratio_Years'][0][0]
            y = self.data.loc[t, 'Dividend_Yields'][0][0]
            y = [v * 100 if v is not None else 0 for v in y]
            ax1.tick_params(axis='x', labelsize=18)
            ax1.tick_params(axis='y', labelsize=18)
            ax1.bar(x, y, width=0.3, alpha=0.6)

            plt.xticks(x, x)
            plt.axhline(y=0, color="black", linestyle=":")
            plt.tight_layout()
            plt.savefig(f'Results/{t}/Images/dividend.jpg')
            plt.close()

            # Create a new report page
            pdf = self.reports[i]
            pdf.add_page()
            pdf.set_font('Courier', style='B', size=50)
            pdf.cell(18, 2, ln=2, align='C', txt=f'VII. Dividend Analysis - {t}')

            pdf.ln(0.5)
            pdf.image(f'Results/{t}/Images/dividend.jpg', w=15, x=2)

            # Question Box 1
            self.question_box(pdf, 10, f'Has the company regularly paid dividend? Has it increased it?')

    def generate_book_value_page(self):
        for i, t in enumerate(self.tickers):
            # Create a new report page
            pdf = self.reports[i]
            pdf.add_page()
            pdf.set_font('Courier', style='B', size=50)
            pdf.cell(18, 2, ln=2, align='C', txt=f'VIII. Book Value Analysis - {t}')

            # Question Box 1
            self.question_box(pdf, 10, "What is the company's book value?")

            # Question Box 2
            self.question_box(pdf, 10, 'Are there any undervalued assets? (Drug patents, undervalued real ' +
                              'estate, shares of another company, durable inventory, etc.)')

    def generate_inventory_page(self):
        for i, t in enumerate(self.tickers):
            # Generate graph 1 and save as image
            fig, ax1 = plt.subplots(1, 1, figsize=(15, 10))
            fig.suptitle(f'Inventory Analysis - {t}', fontsize=24)

            ax1.set_xlabel('Date', fontsize=20)
            ax1.set_ylabel('Inventory', fontsize=20)
            x = self.data.loc[t, 'Inventory_Dates'][0][0]
            y = self.data.loc[t, 'Inventory'][0][0]
            ax1.tick_params(axis='x', labelsize=18)
            ax1.tick_params(axis='y', labelsize=18)
            ax1.bar(x, y, width=0.3, alpha=0.6)

            plt.xticks(x, x)
            plt.axhline(y=0, color="black", linestyle=":")
            plt.tight_layout()
            plt.savefig(f'Results/{t}/Images/inventory.jpg')
            plt.close()

            # Create a new report page
            pdf = self.reports[i]
            pdf.add_page()
            pdf.set_font('Courier', style='B', size=50)
            pdf.cell(18, 2, ln=2, align='C', txt=f'IX. Inventory Analysis - {t}')

            pdf.ln(0.5)
            pdf.image(f'Results/{t}/Images/inventory.jpg', w=16, x=2)

    def generate_sales_page(self):
        for i, t in enumerate(self.tickers):
            # Create a new report page
            pdf = self.reports[i]
            pdf.add_page()
            pdf.set_font('Courier', style='B', size=50)
            pdf.cell(18, 2, ln=2, align='C', txt=f'X. Sales Analysis - {t}')

            # Question Box 1
            self.question_box(pdf, 10, "How are the company's sales distributed?")

            # Question Box 2
            self.question_box(pdf, 10, 'Would a promising product account for a large percent of sales?')

    def generate_profit_margins_page(self):
        for i, t in enumerate(self.tickers):
            # First plot
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            fig.suptitle(f'Profit Margin Analysis - {t}', fontsize=24)

            ax.set_xlabel('Years', fontsize=20)
            ax.set_ylabel('Profit Margin [%]', fontsize=20)
            x = self.data.loc[t, 'Ratio_Years'][0][0]
            y1 = [100 * v for v in self.data.loc[t, 'Gross_Margins'][0][0]]
            y2 = [100 * v for v in self.data.loc[t, 'Operating_Margins'][0][0]]
            y3 = [100 * v for v in self.data.loc[t, 'Pretax_Margins'][0][0]]
            y4 = [100 * v for v in self.data.loc[t, 'Net_Margins'][0][0]]
            ax.plot(x, y1, linewidth=5, label='Gross')
            ax.plot(x, y2, ':', linewidth=5, label='Operating')
            ax.plot(x, y3, '--', linewidth=5, label='Pretax')
            ax.plot(x, y4, '-.', linewidth=5, label='Net')

            ax.tick_params(axis='x', labelsize=18)
            ax.tick_params(axis='y', labelsize=18)
            ax.legend(fontsize=20)

            plt.xticks(x, x)
            plt.tight_layout()
            plt.savefig(f'Results/{t}/Images/profit_margins.jpg')
            plt.close()

            # Create a new report page
            pdf = self.reports[i]
            pdf.add_page()
            pdf.set_font('Courier', style='B', size=50)
            pdf.cell(18, 2, ln=2, align='C', txt=f'XI. Profit Margins Analysis - {t}')

            pdf.ln(0.5)
            pdf.image(f'Results/{t}/Images/profit_margins.jpg', w=16, x=2)

            # Question Box 1
            self.question_box(pdf, 8, "What are the industry's profit margins?")

    def generate_eps_propagation_page(self):
        # Calculate intrinsic value with IntrinsicValueWithEPS
        calculator_eps = IntrinsicValueWithEPS(self.data, span_years=conf.SPAN_YEARS,
                                               desired_returns=conf.DESIRED_RETURNS,
                                               safety_margin=conf.SAFETY_MARGIN)
        data_eps = calculator_eps.run()
        for i, t in enumerate(self.tickers):
            # Plot results of calculation
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            fig.suptitle(f'Intrinsic Value Calculation with EPS Propagation - {t}', fontsize=24)

            ax.set_xlabel('Years', fontsize=20)
            ax.set_ylabel('Price [$]', fontsize=20)

            x = self.price_data[t]['Date'].values
            y = self.price_data[t]['Price'].values
            ax.plot(x, y, linewidth=5, label='Current Price')

            plt.axhline(y=data_eps.loc[t, 'Action_Price_Safe'], linestyle=':', linewidth=5, label='Safe Action Price')
            plt.axhline(y=data_eps.loc[t, 'Future_Price'], linestyle='--', linewidth=5, label='Estimated Future Price')

            ax.tick_params(axis='x', labelsize=18)
            ax.tick_params(axis='y', labelsize=18)
            ax.legend(fontsize=20)

            years = self.price_data[t]['Date'].dt.year.unique()
            plt.xticks([datetime(yr, 1, 1) for yr in years], years)
            plt.tight_layout()
            plt.savefig(f'Results/{t}/Images/eps_propagation.jpg')
            plt.close()

            # Create a new report page
            pdf = self.reports[i]
            pdf.add_page()
            pdf.set_font('Courier', style='B', size=35)
            pdf.cell(18, 1.5, ln=2, align='C', txt=f'XII. Intrinsic Value Calculation with EPS Propagation - {t}')

            pdf.ln(0.25)
            pdf.image(f'Results/{t}/Images/eps_propagation.jpg', w=14, x=2.25)
            pdf.ln(0.25)

            # Collect data to print on page
            eps_table_cols = {'EPS_TTM': 'EPS (TTM)',
                              'EPS_Growth_by_Estimate': 'EPS Growth by Estimate [%]',
                              'EPS_Growth_by_Equity': 'EPS Growth by Equity [%]',
                              'EPS_Growth': 'EPS Growth [%]',
                              'Future_EPS': 'Estimated Future EPS [-]',
                              'Future_PE_by_History': 'Future PE by History [-]',
                              'Future_PE_by_EPS_Growth': 'Future PE by EPS Growth [-]',
                              'Future_PE': 'Estimated Future PE [-]',
                              'Future_Price': 'Estimated Future Price [$]',
                              'Action_Price_Safe': 'Safe Action Price [$]',
                              'Current_Price': 'Current Price [$]'}
            data = pd.DataFrame(data={'Name': [val for val in eps_table_cols.values()],
                                      'Value': [data_eps.loc[t, key] * 100 if '%' in eps_table_cols[key] else data_eps.loc[t, key] for key in eps_table_cols.keys()]})
            # Print data on page
            self.output_df_to_pdf(pdf, data)

    def generate_discounted_cashflow_page(self):
        # Calculate intrinsic value with DCFValuation
        calculator_dcf = DCFValuation(self.data, span_years=conf.SPAN_YEARS, desired_returns=conf.DESIRED_RETURNS,
                                      safety_margin=conf.SAFETY_MARGIN, selling_multiple=conf.SELLING_MULTIPLE)
        data_dcf = calculator_dcf.run()
        for i, t in enumerate(self.tickers):
            # Plot results of calculation
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            fig.suptitle(f'Intrinsic Value Calculation with Discounted Future Cashflows - {t}', fontsize=24)

            ax.set_xlabel('Years', fontsize=20)
            ax.set_ylabel('Price [$]', fontsize=20)

            x = self.price_data[t]['Date'].values
            y = self.price_data[t]['Price'].values
            ax.plot(x, y, linewidth=5, label='Current Price')

            plt.axhline(y=data_dcf.loc[t, 'Action_Price_Safe'], linestyle=':', linewidth=5, label='Safe Action Price')
            plt.axhline(y=data_dcf.loc[t, 'Action_Price'], linestyle='--', linewidth=5, label='Action Price')

            ax.tick_params(axis='x', labelsize=18)
            ax.tick_params(axis='y', labelsize=18)
            ax.legend(fontsize=20)

            years = self.price_data[t]['Date'].dt.year.unique()
            plt.xticks([datetime(yr, 1, 1) for yr in years], years)
            plt.tight_layout()
            plt.savefig(f'Results/{t}/Images/discounted_cashflows.jpg')
            plt.close()

            # Create a new report page
            pdf = self.reports[i]
            pdf.add_page()
            pdf.set_font('Courier', style='B', size=40)
            pdf.multi_cell(18, 1.5, align='C', txt=f'XIII. Intrinsic Value Calculation with DFC - {t}')

            pdf.ln(0.25)
            pdf.image(f'Results/{t}/Images/discounted_cashflows.jpg', w=14, x=2.25)
            pdf.ln(0.25)

            # Collect data to print on page
            dcf_table_cols = {'Starting_Cash_Flow': 'Starting Cash Flow [$]',
                              'Cash_Flow_Growth': 'Cash Flow Growth [%]',
                              f'Cash_Flow_Year_{conf.SPAN_YEARS}': f'Cash Flow Year {conf.SPAN_YEARS} [$]',
                              'Cash_Flow_Sale': 'Cash Flow of Sale [$]',
                              'Current_Cash_and_Investments': 'Current Cash and Investments [$]',
                              'Intrinsic_Value': 'Intrinsic Value [$]',
                              'Action_Price': 'Action Price [$]',
                              'Action_Price_Safe': 'Safe Action Price [$]',
                              'Current_Price': 'Current Price [$]'}
            data = pd.DataFrame(data={'Name': [val for val in dcf_table_cols.values()],
                                      'Value': [data_dcf.loc[t, key] * 100 if '%' in dcf_table_cols[key] else data_dcf.loc[t, key] for key in dcf_table_cols.keys()]})
            # Print data on page
            self.output_df_to_pdf(pdf, data)

    def generate_growth_phase_page(self):
        for i, t in enumerate(self.tickers):
            # Create a new report page
            pdf = self.reports[i]
            pdf.add_page()
            pdf.set_font('Courier', style='B', size=50)
            pdf.cell(18, 2, ln=2, align='C', txt=f'XIV. Growth Phase Analysis - {t}')

            # Question Box 1
            self.question_box(pdf, 17, qs.GROWTH_PHASE)

    def generate_2_minute_drill_page(self):
        for i, t in enumerate(self.tickers):
            # Create a new report page
            pdf = self.reports[i]
            pdf.add_page()
            pdf.set_font('Courier', style='B', size=50)
            pdf.cell(18, 2, ln=2, align='C', txt=f'XV. The Two Minute Drill - {t}')

            # Question Box 1
            self.question_box(pdf, 9, qs.TWD_GENERIC_1)

            # Question Box 2
            self.question_box(pdf, 9, qs.TWD_GENERIC_2)

            for category in qs.TWD_PER_CATEGORY.keys():
                pdf.add_page()
                pdf.set_font('Courier', style='B', size=40)
                pdf.cell(18, 2, ln=2, align='C', txt=f'XV. The Two Minute Drill ({category})- {t}')

                # Question Box 1
                self.question_box(pdf, 20, qs.TWD_PER_CATEGORY[category])

    def save_and_appendix(self):
        # Save reports
        for i, t in enumerate(self.tickers):
            filename = f'Results/{t}/Images/tmp.pdf'
            pdf = self.reports[i]
            pdf.output(filename, 'F')

            merger = PdfFileMerger()
            for file in [filename, 'Common/appendix.pdf']:
                merger.append(file)
            merger.write(f'Results/{t}/Stock Valuation of {t} - {self.ref_date}.pdf')
            merger.close()

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
        values = values[0]
        years = years[0]
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

    @staticmethod
    def output_df_to_pdf(pdf, df):
        # A cell is a rectangular area, possibly framed, which contains some text
        # Set the width and height of cell
        table_cell_width = 6
        table_cell_height = 1
        margin = 4
        pdf.set_font('Arial', 'B', 25)

        # Loop over to print column names
        cols = df.columns
        pdf.cell(margin, table_cell_height, ln=0)
        for col in cols:
            pdf.cell(table_cell_width, table_cell_height, col, align='C', border=1)
        # Line break
        pdf.ln(table_cell_height)
        pdf.cell(margin, table_cell_height, ln=0)
        # Select a font as Arial, regular, 10
        pdf.set_font('Arial', '', 25)
        # Loop over to print each data in the table
        for row in df.itertuples():
            for col in cols:
                value = getattr(row, col)
                if not isinstance(value, str):
                    if '%' in col:
                        value = f'{value:.2f}'
                    elif 'ratio' in col.lower():
                        value = f'{value:.3f}'
                    elif value > 1e6:
                        value = f'{value:.4e}'
                    else:
                        value = f'{value:.4f}'
                pdf.cell(table_cell_width, table_cell_height, value, align='C', border=1)
            pdf.ln(table_cell_height)
            pdf.cell(margin, table_cell_height, ln=0)

    @staticmethod
    def question_box(pdf, length, txt, margin=1):
        pdf.ln(0.5)
        pdf.set_font('Courier', style='', size=30)
        pdf.cell(margin, 0.5, ln=0)
        pdf.multi_cell(16, 0.5, align='L', txt=txt, border='LTR')
        pdf.cell(margin, length - 0.5, ln=0)
        pdf.cell(16, length - 0.5, align='L', txt='', border='LBR', ln=2)
