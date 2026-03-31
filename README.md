# Stock Price Direction Prediction
Predicting whether Apple stock will rise or fall the next day using a Random Forest classifier in Python. 

## Overview
Short-term stock price movements are notoriously difficult to predict 
due to the efficient market hypothesis. This project explores whether 
technical indicators derived from historical price data can be used 
to predict next-day price direction.

## Features Used
- 10 and 50-day moving averages
- Daily return
- Rolling 10-day volatility
- Trading volume

## Results
Achieved 47.9% accuracy on held-out test data — consistent with the 
near-random nature of short-term price movements in efficient markets.
Volume and daily return were identified as the strongest predictive 
features.

## Tools & Libraries
Python, pandas, scikit-learn, yfinance, matplotlib

## What I'd Do Next
- Incorporate news sentiment data using NLP
- Trial XGBoost or LSTM models for comparison
- Expand to a portfolio of stocks rather than a single ticker
