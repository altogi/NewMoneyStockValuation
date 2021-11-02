from data_loader import DataLoader
from action_price_calculator import ActionPriceCalculator
from result_plotter import ResultPlotter


TICKERS = ['AAPL', 'MC.PA', 'MSFT', 'DARK.L', 'NSRGY', 'ADS.DE', 'JNJ']
SPAN_YEARS = 10
DESIRED_RETURNS = 0.15
SAFETY_MARGIN = 0.5

if __name__ == '__main__':
    print('STOCK VALUATION ACCORDING TO NEW MONEY ----> START')
    loader = DataLoader(TICKERS)
    data = loader.run()

    calculator = ActionPriceCalculator(data, span_years=SPAN_YEARS, desired_returns=DESIRED_RETURNS, safety_margin=SAFETY_MARGIN)
    data = calculator.run()

    ResultPlotter(data).run()

    print('STOCK VALUATION ACCORDING TO NEW MONEY <---- END')