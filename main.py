from config import *
from data_loader import DataLoader
from eps_propagation_valuation import IntrinsicValueWithEPS
from discounted_cash_flow_valuation import DCFValuation
from result_plotter import ResultPlotter


if __name__ == '__main__':
    print('STOCK VALUATION ACCORDING TO NEW MONEY ----> START')

    # Load data from Yahoo Finance
    loader = DataLoader(TICKERS, FINANCIAL_MODELING_PREP_API_KEY)
    data = loader.run()

    # Calculate Intrinsic Value Propagating EPS
    calculator_eps = IntrinsicValueWithEPS(data, span_years=SPAN_YEARS, desired_returns=DESIRED_RETURNS, safety_margin=SAFETY_MARGIN)
    data_eps = calculator_eps.run()
    eps_table_cols = ['EPS_TTM', 'EPS_Growth_by_Estimate', 'EPS_Growth_by_Equity', 'EPS_Growth',
                      'Future_EPS', 'Future_PE_by_History', 'Future_PE_by_EPS_Growth', 'Future_PE',
                      'Future_Price', 'Action_Price_Safe', 'Current_Price']

    # Calculate Intrinsic Value with Discounted Future Cashflows
    calculator_dcf = DCFValuation(data, span_years=SPAN_YEARS, desired_returns=DESIRED_RETURNS, safety_margin=SAFETY_MARGIN, selling_multiple=SELLING_MULTIPLE)
    data_dcf = calculator_dcf.run()
    dcf_table_cols = ['Starting_Cash_Flow', 'Cash_Flow_Growth', f'Cash_Flow_Year_{SPAN_YEARS}', 'Cash_Flow_Sale',
                      'Current_Cash_and_Investments', 'Intrinsic_Value', 'Action_Price', 'Action_Price_Safe']

    # Arrange data for plotter
    data = [
        {
            'Data': data_eps,
            'Table_Cols': eps_table_cols,
            'Price_Levels': ['Future_Price', 'Action_Price_Safe'],
            'Price_Level_Names': ['Estimated Future Price of ', 'Safe Action Price of '],
            'Title': 'Valuation Analysis with EPS Propagation'
        },
        {
            'Data': data_dcf,
            'Table_Cols': dcf_table_cols,
            'Price_Levels': ['Action_Price', 'Action_Price_Safe'],
            'Price_Level_Names': ['Intrinsic Value Price of ', 'Safe Action Price of '],
            'Title': 'Valuation Analysis with Discounted Future Cashflows'
        }
    ]
    ResultPlotter(data=data, tickers=TICKERS).run()

    print('STOCK VALUATION ACCORDING TO NEW MONEY <---- END')