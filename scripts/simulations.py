#!/usr/bin/env python
# coding: utf-8

"""
Stock Trading Simulations - Standalone Version
Converted from Jupyter notebook for production use
"""

# IMPORTS
import numpy as np
import pandas as pd
import os
import joblib

# Fin Data Sources
import yfinance as yf
import pandas_datareader as pdr

# Data viz
import plotly.graph_objs as go
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# measure time for ML HyperParams search
import time
from datetime import date

# ML models and utils
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

# Disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'


def load_data():
    """Load the processed data from joblib file"""
    data_dir = os.getcwd() + "/data"
    model_file_name = "new_df.joblib"
    path = os.path.join(data_dir, model_file_name)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}")

    new_df = joblib.load(path)
    print(f"Data loaded successfully. Shape: {new_df.shape}")
    return new_df


def setup_feature_categories(new_df):
    """Define feature categories for analysis"""
    # Growth indicators (but not future growth)
    GROWTH = [g for g in new_df.keys() if (g.find('growth_')==0)&(g.find('future')<0)]

    # OHLCV data
    OHLCV = ['Open','High','Low','Close','Volume']

    # Categorical features
    CATEGORICAL = ['Month', 'Weekday', 'Ticker', 'ticker_type']

    # Future predictions targets
    TO_PREDICT = [g for g in new_df.keys() if (g.find('future')>=0)]

    # Features to drop
    TO_DROP = ['Year','Date', 'Quarter', 'index', 'Capital Gains'] + CATEGORICAL + OHLCV

    # Custom numerical features
    CUSTOM_NUMERICAL = ['SMA10', 'SMA20', 'growing_moving_average', 'high_minus_low_relative',
                       'volatility', 'ln_volume', 'is_div_payout', 'is_stock_split',
                       'Dividends', 'Stock Splits']

    # Technical indicators
    TECHNICAL_INDICATORS = ['adx', 'adxr', 'apo', 'aroon_1','aroon_2', 'aroonosc',
                           'bop', 'cci', 'cmo','dx', 'macd', 'macdsignal', 'macdhist', 'macd_ext',
                           'macdsignal_ext', 'macdhist_ext', 'macd_fix', 'macdsignal_fix',
                           'macdhist_fix', 'mfi', 'minus_di', 'mom', 'plus_di', 'dm', 'ppo',
                           'roc', 'rocp', 'rocr', 'rocr100', 'rsi', 'slowk', 'slowd', 'fastk',
                           'fastd', 'fastk_rsi', 'fastd_rsi', 'trix', 'ultosc', 'willr',
                           'ad', 'adosc', 'obv', 'atr', 'natr', 'ht_dcperiod', 'ht_dcphase',
                           'ht_phasor_inphase', 'ht_phasor_quadrature', 'ht_sine_sine', 'ht_sine_leadsine',
                           'ht_trendmod', 'avgprice', 'medprice', 'typprice', 'wclprice']

    # Technical patterns
    TECHNICAL_PATTERNS = [g for g in new_df.keys() if g.find('cdl')>=0]

    # Macro economic indicators
    MACRO = ['DGS1', 'DGS5', 'DGS10', 'gdppot_us_yoy', 'gdppot_us_qoq',
            'cpi_core_yoy', 'cpi_core_mom', 'FEDFUNDS', 'CSUSHPISA',
            'IRLTLT01DEM156N', 'IRLTLT01GBM156N', 'IRLTLT01JPM156N', '^VIX_Close']

    # Currency pairs
    Currencies = ["EURUSD=X_Close", "GBPUSD=X_Close", "JPY=X_Close", "CNY=X_Close"]

    # All numerical features
    NUMERICAL = GROWTH + TECHNICAL_INDICATORS + TECHNICAL_PATTERNS + CUSTOM_NUMERICAL + MACRO + Currencies

    print(f"Feature categories defined:")
    print(f"  - Growth indicators: {len(GROWTH)}")
    print(f"  - Technical indicators: {len(TECHNICAL_INDICATORS)}")
    print(f"  - Technical patterns: {len(TECHNICAL_PATTERNS)}")
    print(f"  - Custom numerical: {len(CUSTOM_NUMERICAL)}")
    print(f"  - Macro indicators: {len(MACRO)}")
    print(f"  - Currencies: {len(Currencies)}")

    return {
        'GROWTH': GROWTH,
        'OHLCV': OHLCV,
        'CATEGORICAL': CATEGORICAL,
        'TO_PREDICT': TO_PREDICT,
        'TO_DROP': TO_DROP,
        'CUSTOM_NUMERICAL': CUSTOM_NUMERICAL,
        'TECHNICAL_INDICATORS': TECHNICAL_INDICATORS,
        'TECHNICAL_PATTERNS': TECHNICAL_PATTERNS,
        'MACRO': MACRO,
        'Currencies': Currencies,
        'NUMERICAL': NUMERICAL
    }


def get_predictions_correctness(df, to_predict):
    """
    Function to find all predictions (starting from 'pred'), generate is_correct (correctness of each prediction)
    and precision on TEST dataset (assuming there is df["split"] column with values 'train','validation','test'

    Returns 2 lists of features: PREDICTIONS and IS_CORRECT
    """
    PREDICTIONS = [k for k in df.keys() if k.startswith('pred')]
    print(f'Prediction columns found: {len(PREDICTIONS)}')
    print(f'Examples: {PREDICTIONS[:3] if len(PREDICTIONS) >= 3 else PREDICTIONS}')

    # add columns is_correct_
    for pred in PREDICTIONS:
        part1 = pred.split('_')[0]  # first prefix before '_'
        df[f'is_correct_{part1}'] = (df[pred] == df[to_predict]).astype(int)

    # IS_CORRECT features set
    IS_CORRECT = [k for k in df.keys() if k.startswith('is_correct_')]
    print(f'Created is_correct columns: {len(IS_CORRECT)}')

    print('\nPrecision on TEST set for each prediction:')
    # define "Precision" for ALL predictions on a Test dataset (~4 last years of trading)
    for i, column in enumerate(IS_CORRECT):
        prediction_column = PREDICTIONS[i]
        is_correct_column = column
        filter_condition = (df.split == 'test') & (df[prediction_column] == 1)

        if df[filter_condition].shape[0] > 0:
            print(f'Prediction: {prediction_column}')
            value_counts = df[filter_condition][is_correct_column].value_counts()
            value_proportions = value_counts / len(df[filter_condition])
            print(f'  Counts: {dict(value_counts)}')
            print(f'  Proportions: {dict(value_proportions.round(3))}')
            print('-' * 50)
        else:
            print(f'Prediction: {prediction_column} - No positive predictions on test set')
            print('-' * 50)

    return PREDICTIONS, IS_CORRECT


def run_simulation(new_df, predictions_list, to_predict='is_positive_growth_30d_future', investment_amount=50):
    """
    Run trading simulation for all predictions

    Parameters:
    - new_df: DataFrame with all data
    - predictions_list: List of prediction columns
    - to_predict: Target variable name
    - investment_amount: Amount to invest per positive prediction

    Returns:
    - DataFrame with simulation results
    """
    print(f"\n{'='*60}")
    print(f"RUNNING SIMULATION: Investing ${investment_amount} on every positive prediction")
    print(f"Target variable: {to_predict}")
    print(f"{'='*60}")

    sim_results = []

    # Calculate test dataset info
    test_records = len(new_df[new_df.split == 'test'])
    test_days = new_df[new_df.split == 'test']['Date'].nunique()
    print(f"Test dataset: {test_records} records across {test_days} unique days (~4 years)")

    # Iterate over all predictions
    for pred in predictions_list:
        print(f'\nCalculating simulation for prediction: {pred}')

        # Count positive predictions on test set
        positive_predictions = len(new_df[(new_df.split == 'test') & (new_df[pred] == 1)])
        print(f"  Positive predictions: {positive_predictions} out of {test_records} test records")

        if positive_predictions == 0:
            print(f"  Skipping {pred} - no positive predictions on test set")
            continue

        # Prefix for column names
        pred_prefix = pred.split('_')[0]

        # Calculate financial results for each positive prediction
        new_df[f'sim1_gross_rev_{pred_prefix}'] = (new_df[pred] * investment_amount *
                                                  (new_df['growth_future_30d'] - 1))
        new_df[f'sim1_fees_{pred_prefix}'] = -new_df[pred] * investment_amount * 0.002  # 0.2% fees
        new_df[f'sim1_net_rev_{pred_prefix}'] = (new_df[f'sim1_gross_rev_{pred_prefix}'] +
                                                 new_df[f'sim1_fees_{pred_prefix}'])

        # Filter for test set with positive predictions
        filter_condition = (new_df.split == 'test') & (new_df[pred] == 1)

        # Aggregate results
        sim_count_investments = len(new_df[filter_condition])
        sim_gross_rev = new_df[filter_condition][f'sim1_gross_rev_{pred_prefix}'].sum()
        sim_fees = new_df[filter_condition][f'sim1_fees_{pred_prefix}'].sum()
        sim_net_rev = new_df[filter_condition][f'sim1_net_rev_{pred_prefix}'].sum()

        # Calculate metrics
        sim_fees_percentage = -sim_fees / sim_gross_rev if sim_gross_rev > 0 else None
        sim_average_net_revenue = sim_net_rev / sim_count_investments if sim_count_investments > 0 else None

        # Calculate capital requirements
        df_investments_daily = pd.DataFrame(
            new_df[filter_condition].groupby('Date')[pred].count()
        )
        sim_avg_investments_per_day = df_investments_daily[pred].mean()
        sim_q75_investments_per_day = df_investments_daily[pred].quantile(0.75)

        # Capital required (30 days at 75th percentile level)
        sim_capital = 30 * investment_amount * sim_q75_investments_per_day

        # CAGR calculation (4-year period)
        sim_CAGR = ((sim_capital + sim_net_rev) / sim_capital) ** (1/4) if sim_capital > 0 else 1

        # Store results
        sim_results.append((
            pred, sim_count_investments, sim_gross_rev, sim_fees, sim_net_rev,
            sim_fees_percentage, sim_average_net_revenue, sim_avg_investments_per_day,
            sim_capital, sim_CAGR
        ))

        # Print detailed results for predictions with meaningful activity
        if sim_count_investments > 10:  # Only show details for active strategies
            print(f"  Financial Results:")
            print(f"    Investments over 4 years: {sim_count_investments}")
            print(f"    Gross Revenue: ${int(sim_gross_rev)}")

            # Handle None case for fees percentage
            fees_pct_str = f"({int(100 * sim_fees_percentage):.0f}% of gross)" if sim_fees_percentage is not None else "(N/A)"
            print(f"    Fees: ${int(-sim_fees)} {fees_pct_str}")

            print(f"    Net Revenue: ${int(sim_net_rev)}")
            print(f"    Required Capital: ${int(sim_capital)}")
            print(f"    Final Value: ${int(sim_capital + sim_net_rev)}")
            print(f"    CAGR: {sim_CAGR:.3f} ({100 * (sim_CAGR - 1):.1f}%)")
            print(f"    Avg net revenue per trade: ${sim_net_rev / sim_count_investments:.2f}")
            print(f"    Avg investments per day: {sim_avg_investments_per_day:.1f}")

    # Create results DataFrame
    columns = ['prediction', 'sim1_count_investments', 'sim1_gross_rev', 'sim1_fees',
               'sim1_net_rev', 'sim1_fees_percentage', 'sim1_average_net_revenue',
               'sim1_avg_investments_per_day', 'sim1_capital', 'sim1_CAGR']

    df_results = pd.DataFrame(sim_results, columns=columns)
    df_results['sim1_growth_capital_4y'] = (
        (df_results.sim1_net_rev + df_results.sim1_capital) / df_results.sim1_capital
    )

    return df_results


def create_performance_chart(df_results, output_file='performance_chart.html'):
    """Create performance visualization and save to HTML file"""
    print(f"\nCreating performance visualization...")

    # Filter out strategies with no activity
    df_active = df_results[df_results.sim1_count_investments > 10].dropna()

    if len(df_active) == 0:
        print("No active strategies to visualize")
        return None

    fig = px.scatter(
        df_active,
        x='sim1_avg_investments_per_day',
        y='sim1_CAGR',
        size='sim1_growth_capital_4y',
        hover_data=['prediction'],
        title='Compound Annual Growth vs. Daily Trading Frequency',
        labels={
            'sim1_avg_investments_per_day': 'Average Investments per Day',
            'sim1_CAGR': 'Compound Annual Growth Rate',
            'sim1_growth_capital_4y': '4-Year Capital Growth Multiple'
        }
    )

    # Save to HTML file instead of showing
    fig.write_html(output_file)
    print(f"Performance chart saved to: {output_file}")

    return fig


def main():
    """Main execution function"""
    try:
        print("Starting Stock Trading Simulation Analysis")
        print("=" * 50)

        # Load data
        new_df = load_data()

        # Setup feature categories
        features = setup_feature_categories(new_df)

        # Clean up data
        if 'index_x' in new_df.columns:
            new_df = new_df.rename(columns={'index_x': 'index'})
        if 'index_y' in new_df.columns:
            new_df = new_df.drop('index_y', axis=1)

        # Define target variable
        to_predict = 'is_positive_growth_30d_future'

        # Get predictions and their correctness
        predictions_list, is_correct_list = get_predictions_correctness(new_df, to_predict)

        # Run simulation
        df_simulation_results = run_simulation(new_df, predictions_list, to_predict)

        # Display final results
        print(f"\n{'='*60}")
        print("SIMULATION RESULTS SUMMARY")
        print(f"{'='*60}")

        # Sort by CAGR descending
        df_sorted = df_simulation_results.sort_values('sim1_CAGR', ascending=False)

        # Display top performers
        print("\nTop 10 Performing Strategies (by CAGR):")
        top_strategies = df_sorted.head(10)

        for _, row in top_strategies.iterrows():
            if row['sim1_count_investments'] > 0:
                print(f"  {row['prediction']}: CAGR={row['sim1_CAGR']:.3f} "
                      f"({100*(row['sim1_CAGR']-1):.1f}%), "
                      f"Trades={int(row['sim1_count_investments'])}, "
                      f"Net=${int(row['sim1_net_rev'])}")

        # Create and save performance chart (no more fig.show())
        fig = create_performance_chart(df_simulation_results)

        # Save results
        output_file = 'simulation_results.csv'
        df_simulation_results.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

        return df_simulation_results, new_df

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    results_df, data_df = main()
