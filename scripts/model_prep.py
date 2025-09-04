#!/usr/bin/env python
# coding: utf-8

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

# Measure time for ML HyperParams search
import time
from datetime import date

# ML models and utils
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

# Disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None

# Load data
data_dir = os.getcwd() + "/data"
file_name = "stocks_indices_macros_df.parquet"
print("About to load parquet file...")
df = pd.read_parquet(os.path.join(data_dir, file_name))
print(f"Loaded successfully: {df.shape}")

# Define feature groups
GROWTH = [g for g in df.keys() if (g.find('growth_')==0)&(g.find('future')<0)]
OHLCV = ['Open','High','Low','Close','Close_x','Volume']
CATEGORICAL = ['Month', 'Weekday', 'Ticker', 'ticker_type']

# Clean up column names
df = df.rename(columns = {'Month_x': 'Month'})
df = df.drop('Month_y', axis=1)

# Define prediction targets
TO_PREDICT = [g for g in df.keys() if (g.find('future')>=0)]

# Define features to drop
TO_DROP = ['Year','Date', 'Quarter', 'index', 'index_x', 'index_y', 'Close_y', 'Capital Gains'] + CATEGORICAL + OHLCV

# Create custom numerical features
df['ln_volume'] = df.Volume.apply(lambda x: np.log(x+ 1e-6))
df['is_div_payout'] = (df.Dividends>0).astype(int)
df['is_stock_split'] = (df['Stock Splits']>0).astype(int)

# Define feature groups
CUSTOM_NUMERICAL = ['SMA10', 'SMA20', 'growing_moving_average', 'high_minus_low_relative','volatility', 'ln_volume', 'is_div_payout', 'is_stock_split', 'Dividends', 'Stock Splits']

TECHNICAL_INDICATORS = ['adx', 'adxr', 'apo', 'aroon_1','aroon_2', 'aroonosc',
 'bop', 'cci', 'cmo','dx', 'macd', 'macdsignal', 'macdhist', 'macd_ext',
 'macdsignal_ext', 'macdhist_ext', 'macd_fix', 'macdsignal_fix',
 'macdhist_fix', 'mfi', 'minus_di', 'mom', 'plus_di', 'dm', 'ppo',
 'roc', 'rocp', 'rocr', 'rocr100', 'rsi', 'slowk', 'slowd', 'fastk',
 'fastd', 'fastk_rsi', 'fastd_rsi', 'trix', 'ultosc', 'willr',
 'ad', 'adosc', 'obv', 'atr', 'natr', 'ht_dcperiod', 'ht_dcphase',
 'ht_phasor_inphase', 'ht_phasor_quadrature', 'ht_sine_sine', 'ht_sine_leadsine',
 'ht_trendmod', 'avgprice', 'medprice', 'typprice', 'wclprice']

TECHNICAL_PATTERNS = [g for g in df.keys() if g.find('cdl')>=0]

MACRO = ['DGS1', 'DGS5', 'DGS10', 'gdppot_us_yoy', 'gdppot_us_qoq',
        'cpi_core_yoy', 'cpi_core_mom', 'FEDFUNDS', 'CSUSHPISA',
       'IRLTLT01DEM156N', 'IRLTLT01GBM156N', 'IRLTLT01JPM156N', '^VIX_Close']

Currencies = ["EURUSD=X_Close", "GBPUSD=X_Close", "JPY=X_Close", "CNY=X_Close"]

NUMERICAL = GROWTH + TECHNICAL_INDICATORS + TECHNICAL_PATTERNS + CUSTOM_NUMERICAL + MACRO + Currencies

# Create dummy variables
# df.loc[:,'Month'] = df['Date'].dt.month_name().astype('string')
# df.loc[:,'Weekday'] = df['Date'].dt.day_name().astype('string')
df['Month'] = df['Date'].dt.month_name().astype('string')
df['Weekday'] = df['Date'].dt.day_name().astype('string')
print("Month and Weekday columns created successfully")


dummy_variables = pd.get_dummies(df[CATEGORICAL], dtype='int32')
DUMMIES = dummy_variables.keys().to_list()
print(f"Created {len(DUMMIES)} dummy variables")

# Concatenate dummy variables with original DataFrame
df_with_dummies = pd.concat([df, dummy_variables], axis=1)
print("Concatenation completed")
print(f"df_with_dummies shape: {df_with_dummies.shape}")
print(f"Memory usage: {df_with_dummies.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

def temporal_split(df, min_date, max_date, train_prop=0.7, val_prop=0.15, test_prop=0.15):
    """
    Splits a DataFrame into three buckets based on the temporal order of the 'Date' column.
    """
    train_end = min_date + pd.Timedelta(days=(max_date - min_date).days * train_prop)
    val_end = train_end + pd.Timedelta(days=(max_date - min_date).days * val_prop)

    split_labels = []
    for date in df['Date']:
        if date <= train_end:
            split_labels.append('train')
        elif date <= val_end:
            split_labels.append('validation')
        else:
            split_labels.append('test')

    df['split'] = split_labels
    return df

print("Getting min/max dates...")
min_date_df = df_with_dummies.Date.min()
max_date_df = df_with_dummies.Date.max()
print(f"Date range: {min_date_df} to {max_date_df}")

# Apply temporal split
min_date_df = df_with_dummies.Date.min()
max_date_df = df_with_dummies.Date.max()
print("Starting temporal split...")
df_with_dummies = temporal_split(df_with_dummies, min_date=min_date_df, max_date=max_date_df)
print("Temporal split completed")
# Create clean copy
print("Creating clean dataframe copy...")
new_df = df_with_dummies.copy()
print("Clean copy completed")

# Prepare feature sets
print("Preparing feature sets...")
features_list = NUMERICAL + DUMMIES
print(f"Features list length: {len(features_list)}")
to_predict = 'is_positive_growth_30d_future'

# Create data splits
print("Creating train split...")
train_df = new_df[new_df.split.isin(['train'])].copy(deep=True)
print("Creating valid split...")
valid_df = new_df[new_df.split.isin(['validation'])].copy(deep=True)
print("Creating train_valid split...")
train_valid_df = new_df[new_df.split.isin(['train','validation'])].copy(deep=True)
print("Creating test split...")
test_df = new_df[new_df.split.isin(['test'])].copy(deep=True)
print("Data splits completed")

# Prepare feature matrices
print("Creating feature matrices...")
print(f"About to create X_train with {len(features_list+[to_predict])} columns...")
X_train = train_df[features_list+[to_predict]]
print("X_train created")
print("Creating X_valid...")
X_valid = valid_df[features_list+[to_predict]]
print("X_valid created")
print("Creating X_train_valid...")
X_test = test_df[features_list+[to_predict]]
print("X_train_valid created")
print("Creating X_test...")
X_train_valid = train_valid_df[features_list+[to_predict]]
print("X_test created")
print("Creating X_all...")
X_all = new_df[features_list+[to_predict]].copy(deep=True)
print("X_all created")

def clean_dataframe_from_inf_and_nan(df: pd.DataFrame):
    """Clean dataframe by replacing inf values with NaN and filling NaN with 0"""
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

# Clean datasets
print("Starting data cleaning...")
X_train = clean_dataframe_from_inf_and_nan(X_train)
X_valid = clean_dataframe_from_inf_and_nan(X_valid)
X_train_valid = clean_dataframe_from_inf_and_nan(X_train_valid)
X_test = clean_dataframe_from_inf_and_nan(X_test)
X_all = clean_dataframe_from_inf_and_nan(X_all)
print("datasets cleaned")

# Separate targets from features
y_train = X_train[to_predict]
y_valid = X_valid[to_predict]
y_train_valid = X_train_valid[to_predict]
y_test = X_test[to_predict]
y_all = X_all[to_predict]

del X_train[to_predict]
del X_valid[to_predict]
del X_train_valid[to_predict]
del X_test[to_predict]
del X_all[to_predict]

# Generate manual predictions
new_df['pred0_manual_cci'] = (new_df.cci>200).astype(int)
new_df['pred1_manual_prev_g1'] = (new_df.growth_30d>1).astype(int)
new_df['pred2_manual_prev_g1_and_snp'] = ((new_df['growth_30d'] > 1) & (new_df['growth_^GSPC_30d'] > 1)).astype(int)
new_df['pred3_manual_dgs10_5'] = ((new_df['DGS10'] <= 4) & (new_df['DGS5'] <= 2)).astype(int)
new_df['pred4_manual_dgs10_fedfunds'] = ((new_df['DGS10'] > 4) & (new_df['FEDFUNDS'] <= 4.795)).astype(int)

def get_predictions_correctness(df: pd.DataFrame, to_predict: str):
    """Calculate prediction correctness and precision on test set"""
    PREDICTIONS = [k for k in df.keys() if k.startswith('pred')]

    # Add correctness columns
    for pred in PREDICTIONS:
        part1 = pred.split('_')[0]
        df[f'is_correct_{part1}'] = (df[pred] == df[to_predict]).astype(int)

    IS_CORRECT = [k for k in df.keys() if k.startswith('is_correct_')]
    return PREDICTIONS, IS_CORRECT

PREDICTIONS, IS_CORRECT = get_predictions_correctness(df=new_df, to_predict=to_predict)

# Decision Tree Classifier functions
def fit_decision_tree(X, y, max_depth=20):
    """Fit decision tree classifier"""
    print(f"INSIDE fit_decision_tree function with max_depth={max_depth}")
    print(f"  -> Fitting Decision Tree with max_depth={max_depth} on {X.shape[0]} samples...")
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X, y)
    return clf, X.columns

print(f"About to train with X_train_valid shape: {X_train_valid.shape}")
print(f"y_train_valid shape: {y_train_valid.shape}")
print(f"X_train_valid dtypes: {X_train_valid.dtypes.value_counts()}")
print("Calling fit_decision_tree now...")
# Train decision trees with different depths
clf_13, train_columns = fit_decision_tree(X=X_train_valid, y=y_train_valid, max_depth=13)
y_pred_all = clf_13.predict(X_all)
new_df['pred5_clf_13'] = y_pred_all

# Best hyperparameters from previous tuning
precision_by_depth = {1: 0.5475, 2: 0.4458, 3: 0.4458, 4: 0.5112, 5: 0.4975, 6: 0.5292, 7: 0.5189, 8: 0.5425, 9: 0.5361, 10: 0.5494, 11: 0.5302, 12: 0.5378, 13: 0.5463, 14: 0.5392, 15: 0.5417, 16: 0.538, 17: 0.5457, 18: 0.5317, 19: 0.5383, 20: 0.535}

clf_17, train_columns = fit_decision_tree(X=X_train_valid, y=y_train_valid, max_depth=17)
y_pred_all = clf_17.predict(X_all)
new_df['pred6_clf_17'] = y_pred_all

# Best depth model
best_depth = 10
best_precision = precision_by_depth[best_depth]

clf_best, train_columns = fit_decision_tree(X=X_train_valid, y=y_train_valid, max_depth=best_depth)
y_pred_clf_best = clf_best.predict(X_all)
new_df['pred6_clf_best'] = y_pred_clf_best

# Random Forest
# Pre-calculated best precision matrix
best_precision_matrix_random_forest = {(5, 10): 0.5573, (5, 50): 0.5593, (5, 100): 0.5615, (5, 200): 0.5649, (7, 10): 0.5185, (7, 50): 0.5211, (7, 100): 0.526, (7, 200): 0.5442, (9, 10): 0.5377, (9, 50): 0.5226, (9, 100): 0.5296, (9, 200): 0.5399, (11, 10): 0.5334, (11, 50): 0.5202, (11, 100): 0.5279, (11, 200): 0.5237, (13, 10): 0.5411, (13, 50): 0.5422, (13, 100): 0.5436, (13, 200): 0.5445, (15, 10): 0.5342, (15, 50): 0.5268, (15, 100): 0.5321, (15, 200): 0.5279, (17, 10): 0.5438, (17, 50): 0.5316, (17, 100): 0.5303, (17, 200): 0.5427, (19, 10): 0.5509, (19, 50): 0.5406, (19, 100): 0.539, (19, 200): 0.5376}

# Best Random Forest hyperparameters
rf_best_n_estimators = 100
rf_best_max_depth = 13

# Train Random Forest
rf_best = RandomForestClassifier(n_estimators=rf_best_n_estimators, max_depth=rf_best_max_depth, random_state=42, n_jobs=-1)
rf_best = rf_best.fit(X_train_valid, y_train_valid)

def tpr_fpr_dataframe(y_true, y_pred, only_even=False):
    """Calculate TPR/FPR for different thresholds"""
    scores = []

    if only_even == False:
        thresholds = np.linspace(0, 1, 101)
    else:
        thresholds = np.linspace(0, 1, 51)

    for t in thresholds:
        actual_positive = (y_true == 1)
        actual_negative = (y_true == 0)
        predict_positive = (y_pred >= t)
        predict_negative = (y_pred < t)

        tp = (predict_positive & actual_positive).sum()
        tn = (predict_negative & actual_negative).sum()
        fp = (predict_positive & actual_negative).sum()
        fn = (predict_negative & actual_positive).sum()

        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0

        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0

        if precision + recall > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0

        accuracy = (tp + tn) / (tp + tn + fp + fn)

        scores.append((t, tp, fp, fn, tn, precision, recall, accuracy, f1_score))

    columns = ['threshold', 'tp', 'fp', 'fn', 'tn', 'precision', 'recall', 'accuracy', 'f1_score']
    df_scores = pd.DataFrame(scores, columns=columns)

    df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
    df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)

    return df_scores

# Generate threshold-based predictions for Decision Tree
y_pred_all = clf_best.predict_proba(X_all)
y_pred_all_class1 = [k[1] for k in y_pred_all]
y_pred_all_class1_array = np.array(y_pred_all_class1)

new_df['proba_pred8'] = y_pred_all_class1_array
new_df['pred8_clf_second_best_rule_84'] = (y_pred_all_class1_array >= 0.84).astype(int)
new_df['proba_pred9'] = y_pred_all_class1_array
new_df['pred9_clf_second_best_rule_92'] = (y_pred_all_class1_array >= 0.92).astype(int)

# Generate threshold-based predictions for Random Forest
y_pred_all = rf_best.predict_proba(X_all)
y_pred_all_class1 = [k[1] for k in y_pred_all]
y_pred_all_class1_array = np.array(y_pred_all_class1)

new_df['proba_pred10'] = y_pred_all_class1_array
new_df['pred10_rf_best_rule_55'] = (y_pred_all_class1_array >= 0.55).astype(int)
new_df['proba_pred11'] = y_pred_all_class1_array
new_df['pred11_rf_best_rule_65'] = (y_pred_all_class1_array >= 0.65).astype(int)

# Final update of predictions and correctness
print("Final update of predictions and correctness...")
PREDICTIONS, IS_CORRECT = get_predictions_correctness(new_df, to_predict=to_predict)
print(f"Total predictions generated: {len(PREDICTIONS)}")

# Save the final dataframe
print("Saving final dataframe...")
filename = "new_df.joblib"
path = os.path.join(data_dir, filename)
joblib.dump(new_df, path)
print(f"Dataframe saved to: {path}")
print("Model preparation completed successfully!")
