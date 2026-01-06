import yfinance as yf
import pandas as pd

def download_stock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Baixa dados históricos de ações do Yahoo Finance.

    Args:
        symbol (str): Símbolo da ação (ex: 'DIS').
        start_date (str): Data de início no formato 'YYYY-MM-DD'.
        end_date (str): Data de fim no formato 'YYYY-MM-DD'.

    Returns:
        pd.DataFrame: DataFrame com os dados históricos da ação.
    """
    df = yf.download(symbol, start=start_date, end=end_date)
    return df

if __name__ == '__main__':
    # Exemplo de uso
    symbol = 'PETR4.SA'  # Exemplo de ação brasileira
    start_date = '2018-01-01'
    end_date = '2024-07-20'
    stock_data = download_stock_data(symbol, start_date, end_date)
    print(stock_data.head())

