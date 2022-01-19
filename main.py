from config import *
from report_generator import ReportGenerator


if __name__ == '__main__':
    print('AUTOMATIC STOCK REPORT GENERATION ----> START')

    ReportGenerator(TICKERS, FINANCIAL_MODELING_PREP_API_KEY).run()

    print('AUTOMATIC STOCK REPORT GENERATION <---- END')