import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing

matplotlib.use("Agg")
import datetime
import torch

from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.env.lxc_env_stocktrading import lxcStockTradingEnv
from finrl.model.models import DRLAgent
from finrl.trade.backtest import backtest_stats as BackTestStats
from stable_baselines3 import A2C

def train_one():
    """
    train an agent
    """
    print("==============Start Fetching Data===========")
    
    df = YahooDownloader(
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        ticker_list=config.DOW_30_TICKER,
    ).fetch_data()
    df.to_csv('/result',index=False)