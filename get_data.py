#!/usr/bin/env python
"""
Stock Data Collection Script
Fetches stock prices, economic indices, and macro indicators
"""

import numpy as np
import pandas as pd
import requests
from io import StringIO
from fredapi import Fred
from dotenv import load_dotenv
import os
import yfinance as yf
import pandas_datareader as pdr
import time
from datetime import date

# Disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None


def fetch_biggest_companies():
    """Fetch list of biggest companies from stockanalysis.com"""
    url = f"https://stockanalysis.com/list/biggest-companies/"
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/58.0.3029.110 Safari/537.3'
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        html_io = StringIO(response.text)
        table = pd.read_html(html_io)
        df = table[0]
        symbols = df["Symbol"].to_list()[:100]

        if not table:
            raise ValueError(f"No table found")

        # Fix BRK.B symbol format
        if 'BRK.B' in symbols:
            symbols[symbols.index('BRK.B')] = 'BRK-B'
        
        return symbols

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return []
    except ValueError as ve:
        print(f"Data error: {ve}")
        return []
    except Exception as ex:
        print(f"Unexpected error: {ex}")
        return []


def setup_tickers():
    """Setup all ticker lists"""
    symbols = fetch_biggest_companies()
    
    tariff_stocks = ["FINV", "TSM", "LITE", "VOYA", "KO", "BUG", "WEC"]
    duplicates = set(tariff_stocks).intersection(set(symbols))
    for dup in duplicates:
        symbols.remove(dup)
    
    commodity_funds = ["GLD", "USO", "MOO", "WELL", "BND", "REMX", "SLV"]
    world_indices = ['^GSPC', '^DJI', '^STOXX', '^GDAXI', '^FTSE', '^N225', '^VIX']
    commodities = ['GC=F', 'HG=F', 'BZ=F', 'S=F', 'BTC-USD']
    currencies = ["EURUSD=X", "GBPUSD=X", "JPY=X", "CNY=X"]
    bonds = ['IRLTLT01DEM156N', 'IRLTLT01GBM156N', 'IRLTLT01JPM156N']
    
    tickers = symbols + tariff_stocks + commodity_funds
    indices = world_indices + commodities + currencies
    macros = ['GDPPOT', 'CPILFESL', 'FEDFUNDS', 'DGS1', 'DGS5', 'DGS10', 'CSUSHPISA']
    
    return tickers, indices, macros, bonds, symbols, tariff_stocks


def get_growth_df(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Calculate growth indicators for a dataframe"""
    for i in [1, 3, 7, 30, 90, 365]:
        df['growth_' + prefix + '_' + str(i) + 'd'] = df['Close'] / df['Close'].shift(i)
    
    GROWTH_KEYS = [k for k in df.keys() if k.startswith('growth')]
    return df[GROWTH_KEYS]


def fetch_stock_data(tickers, start_date, end_date, symbols, tariff_stocks):
    """Fetch stock data for all tickers"""
    stocks_df = pd.DataFrame({'A': []})

    for i, t in enumerate(tickers):
        print(f"{i+1}/{len(tickers)} Processing ticker: {t}")
        try:
            historyPrices = yf.Ticker(t).history(start=start_date, interval="1d")
            
            if historyPrices.empty:
                print(f"No data found for {t}")
                continue
                
            if t in symbols:
                historyPrices['ticker_type'] = 'biggest_company'
            elif t in tariff_stocks:
                historyPrices['ticker_type'] = 'tariff'
            else:
                historyPrices['ticker_type'] = 'commodity_fund'
            
            historyPrices['Ticker'] = t
            historyPrices['Year'] = historyPrices.index.year
            historyPrices['Month'] = historyPrices.index.month
            historyPrices['Weekday'] = historyPrices.index.weekday
            historyPrices['Date'] = historyPrices.index.date
            
            # Historical returns
            for period in [1, 3, 7, 30, 90, 365]:
                historyPrices[f'growth_{period}d'] = historyPrices['Close'] / historyPrices['Close'].shift(period)
            
            historyPrices['growth_future_30d'] = historyPrices['Close'].shift(-30) / historyPrices['Close']
            
            # Technical indicators
            historyPrices['SMA10'] = historyPrices['Close'].rolling(10).mean()
            historyPrices['SMA20'] = historyPrices['Close'].rolling(20).mean()
            historyPrices['growing_moving_average'] = np.where(historyPrices['SMA10'] > historyPrices['SMA20'], 1, 0)
            historyPrices['high_minus_low_relative'] = (historyPrices.High - historyPrices.Low) / historyPrices['Close']
            
            # 30d rolling volatility
            historyPrices['volatility'] = historyPrices['Close'].rolling(30).std() * np.sqrt(252)
            
            # Target variable
            historyPrices['is_positive_growth_30d_future'] = np.where(historyPrices['growth_future_30d'] > 1, 1, 0)
            
            # Sleep to avoid overloading API
            time.sleep(1)
            
            if stocks_df.empty:
                stocks_df = historyPrices
            else:
                stocks_df = pd.concat([stocks_df, historyPrices], ignore_index=True)
                
        except Exception as e:
            print(f"Error processing {t}: {e}")
            continue
    
    return stocks_df


def fetch_indices_data(indices, start_date):
    """Fetch economic indices data"""
    to_merge = []
    
    for i, t in enumerate(indices):
        print(f"{i+1}/{len(indices)} Processing index: {t}")
        try:
            econ = yf.Ticker(t).history(start=start_date, interval="1d")
            
            if econ.empty:
                print(f"No data found for {t}")
                continue
                
            econ.index = econ.index.date

            if t not in ["EURUSD=X", "GBPUSD=X", "JPY=X", "CNY=X"] and t != '^VIX':
                to_merge.append(get_growth_df(econ, t))
            else:
                to_merge.append(econ.rename(columns={"Close": t + "_Close"})[[t + "_Close"]])
            
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing {t}: {e}")
            continue
    
    # Merge all indices data
    if not to_merge:
        return pd.DataFrame()
    
    merged = to_merge[0]
    for j in range(1, len(to_merge)):
        merged = pd.merge(
            merged,
            to_merge[j],
            left_index=True,
            right_index=True,
            how="left",
            validate="one_to_one",
        )
    
    return merged


def fetch_macro_data(start_date, bonds):
    """Fetch macroeconomic data from FRED"""
    load_dotenv()
    fred_api_key = os.getenv("FRED_API_KEY")
    
    if not fred_api_key:
        print("Warning: FRED_API_KEY not found in environment variables")
        return pd.DataFrame()
    
    try:
        # GDP Potential
        gdppot = pdr.DataReader("GDPPOT", "fred", start=start_date).reset_index()
        gdppot['gdppot_us_yoy'] = gdppot.GDPPOT / gdppot.GDPPOT.shift(4) - 1
        gdppot['gdppot_us_qoq'] = gdppot.GDPPOT / gdppot.GDPPOT.shift(1) - 1
        gdppot["Quarter"] = gdppot.DATE.dt.to_period('Q').dt.start_time
        gdppot_to_merge = gdppot[["Quarter", "gdppot_us_yoy", "gdppot_us_qoq"]]

        # Core CPI
        cpilfesl = pdr.DataReader("CPILFESL", "fred", start=start_date).reset_index()
        cpilfesl['cpi_core_yoy'] = cpilfesl.CPILFESL / cpilfesl.CPILFESL.shift(12) - 1
        cpilfesl['cpi_core_mom'] = cpilfesl.CPILFESL / cpilfesl.CPILFESL.shift(1) - 1
        cpilfesl["Month"] = cpilfesl.DATE.dt.to_period('M').dt.start_time
        cpilfesl_to_merge = cpilfesl[["Month", "cpi_core_yoy", "cpi_core_mom"]]

        # Fed Funds Rate
        fedfunds = pdr.DataReader("FEDFUNDS", "fred", start=start_date).reset_index()
        fedfunds["Month"] = fedfunds.DATE.dt.to_period('M').dt.start_time
        fedfunds.set_index('DATE', inplace=True)

        # Case-Shiller Housing Index
        caseshiller = pdr.DataReader('CSUSHPISA', "fred", start=start_date).reset_index()
        caseshiller["Month"] = caseshiller.DATE.dt.to_period('M').dt.start_time
        caseshiller.set_index('DATE', inplace=True)

        # Treasury rates
        dgs1 = pdr.DataReader("DGS1", "fred", start=start_date).reset_index()
        dgs5 = pdr.DataReader("DGS5", "fred", start=start_date).reset_index()
        dgs10 = pdr.DataReader("DGS10", "fred", start=start_date).reset_index()

        # International bonds
        bonds_data = []
        for bond in bonds:
            print(f"Processing bond: {bond}")
            b_data = pdr.DataReader(bond, 'fred', start=start_date).reset_index()
            b_data['Month'] = b_data.DATE.dt.to_period('M').dt.start_time
            b_data.set_index('DATE', inplace=True)
            bonds_data.append(b_data)

        # Merge treasury rates
        macros_df = dgs1
        for df in [dgs5, dgs10]:
            macros_df = pd.merge(macros_df, df, on='DATE', how='inner', validate="one_to_one")

        macros_df["Quarter"] = macros_df.DATE.dt.to_period('Q').dt.start_time
        macros_df["Month"] = macros_df.DATE.dt.to_period('M').dt.start_time

        # Merge quarterly data
        macros_df = pd.merge(macros_df, gdppot_to_merge, on="Quarter", how="left", validate="many_to_one")
        
        # Merge monthly data
        macros_df = pd.merge(macros_df, cpilfesl_to_merge, on="Month", how="left", validate="many_to_one")
        macros_df = pd.merge(macros_df, fedfunds, on="Month", how="left", validate="many_to_one")
        macros_df = pd.merge(macros_df, caseshiller, on="Month", how="left", validate="many_to_one")

        # Merge bond data
        for bond_df in bonds_data:
            macros_df = pd.merge(macros_df, bond_df, on="Month", how="left", validate="many_to_one")

        # Forward fill missing values
        macros_df = macros_df.ffill()
        
        return macros_df
        
    except Exception as e:
        print(f"Error fetching macro data: {e}")
        return pd.DataFrame()


def save_data(stocks_df, econ_indices_df, macros_df):
    """Save data to parquet files"""
    data_dir = os.path.join(os.getcwd(), "data")
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    datasets = {
        "stocks_df": stocks_df,
        "econ_indices_df": econ_indices_df,
        "macros_df": macros_df
    }
    
    for name, df in datasets.items():
        if not df.empty:
            file_path = os.path.join(data_dir, f"{name}.parquet")
            try:
                df.to_parquet(file_path, compression="brotli")
                print(f"Saved {name} to {file_path}")
            except Exception as e:
                print(f"Error saving {name}: {e}")
        else:
            print(f"Warning: {name} is empty, not saving")


def main():
    """Main execution function"""
    print("Starting data collection process...")
    
    # Setup date range
    end = date.today()
    start = date(year=end.year-50, month=end.month, day=end.day)
    print(f'Date range: {start} to {end}')
    
    # Setup tickers
    tickers, indices, macros, bonds, symbols, tariff_stocks = setup_tickers()
    print(f"Processing {len(tickers)} stocks, {len(indices)} indices, {len(bonds)} bonds")
    
    # Fetch data
    print("\n=== Fetching Stock Data ===")
    stocks_df = fetch_stock_data(tickers, start, end, symbols, tariff_stocks)
    
    print("\n=== Fetching Indices Data ===")
    econ_indices_df = fetch_indices_data(indices, start)
    
    print("\n=== Fetching Macro Data ===")
    macros_df = fetch_macro_data(start, bonds)
    
    # Save data
    print("\n=== Saving Data ===")
    save_data(stocks_df, econ_indices_df, macros_df)
    
    print("Data collection completed!")
    return stocks_df, econ_indices_df, macros_df


if __name__ == "__main__":
    stocks_df, econ_indices_df, macros_df = main()
