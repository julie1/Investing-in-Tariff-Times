# Investing-in-Tariff-Times
This repository investigates a Machine Learning strategy for investment in the tariff economy.
This project was implemented for stock markets analytics zoomcamp, a free course about developing
Machine Learning methods for investment.  Large tariffs have been imposed on the world economy in
2025.  We wish to use various macro-economic factors and stock market indices to determine an investment
strategy that may produce positive returns despite the handicapping effect of tariffs.

## Project overview

In order to implement our strategy we must first obtain datasets consisting of stocks, economic indices,
and macro-economic variables.  Then the datasets need to be converted to a form suitable for Machine Learning (ML).  
The stock dataset will be augmented by the financial analysis provided by the Ta-lib library.  Categorical 
data needs to be converted to numerical data through dummy variables which are included in the features provided to 
the ML model.  The three datasets are merged into one.  Then we apply  ML models to determine probability threshholds for trading

## Datasets

There are three datasets to start.
- **stocks_df** data downloaded from yfinance consists of the
- biggest 100 companies from https://stockanalysis.com/list/biggest-companies
- seven of the largest commodity funds
- seven tariff stocks with four recommended from Seeking Alpha
    https://seekingalpha.com/article/4809784-the-art-of-the-tariff-4-stocks-to-buy-the-dipand three others 
- **econ_indices_df** consists of seven world indices, five commodities, and four currency exchange rates with USD with data from yfinance
- **macros_df** consists of seven macro-economical indices plus three international bond rates obtained from FRED
   
## Dataset Transformations
 - we truncate the data with start date: 2000-01-01
 - Technical indicators from the Ta-lib library are added to the stocks_df dataset
 - The three datasets are merged into the stocks_indices_macros_df which is saved
 - Feature sets are defined including Dummies, Numerical, and To Predict which includes the 30 day growth
   of the asset and whether the growth is positive
 - We used a temporal split function to generate training, validation, and test sets for Machine Learning applications

##  Modeling
- Decision Tree Classification, Random Forest, and a Deep Learning model were explored in the notebooks, but only the first
 two were included in our pipeline
- New predictions were generated from the ML models

##  Simulations
- We did an approximate simulation to determine the Compound Annual Growth Rate (CAGR) and the average number of trades per day for each prediction where each trade is $50
- We started a more complicated simulation but did not include that in our pipeline

## Code

The code for the trading pipeline is in the  [`scripts`](scripts/) folder:

- [`get_data.py`](scripts/get_data.py) - Downloads data
- [`transform_data`](scripts/transform_data.py) - the truncation, adding technical indicators, and merging of data
- [`model_prep.py`](scripts/model_prep.py) - the modeling 
- [`simulations.py`](scripts/simulations.py) - simulations
- [`latest_trades.py`](scripts/latest_trades.py) - show the latest trades and save to a csv file
- [`pipeline.py`](scripts/pipeline.py) - run the five above scripts

## Notebooks

The notebooks are in the [`notebooks`](notebooks/) folder and include more visualization graphs and experimentation

## How to Run 

- create a python environment
- install the requirements from the requirements.txt file
- obtain a FRED_API_KEY from https://fred.stlouisfed.org/ and put in a .env file
- the run:
 ```bash
    python pipeline.py
 ```
## Acknowledgements 

 We thank Ivan Brigida for an interesting and informative course.  Much of our code here was adapted 
 from https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp/blob/main/05-deployment-and-automation/%5B2025%5D_Module_05_Advanced_Strategies_And_Simulation.ipynb   We hope to make further improvements to this 
 project.