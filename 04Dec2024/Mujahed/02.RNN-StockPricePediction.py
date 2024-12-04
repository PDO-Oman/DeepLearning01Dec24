# 01. Load Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf

# 02: Download Stock Data (e.g., Apple)
stock_data = yf.download('AAPL', start='2010-01-01', end='2024-01-01')
