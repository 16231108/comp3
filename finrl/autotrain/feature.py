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

def train_one(data_path):
    """
    train an agent
    """
    print("==============Start Fetching Data===========")
    
    df = pd.read_csv(data_path)
    
    #names=["date","open","high","low","close","volume","tic","day",]
    #df = pd.read_csv("./" + config.DATA_SAVE_DIR + "/" + "20210315-07h382" + ".csv",index_col=0)
    print('GPU is :',torch.cuda.is_available())
    #df = pd.read_csv("./" + config.DATA_SAVE_DIR + "/" + "20210315-08h17" + ".csv", index_col=0)
    #print(df)
    print("==============Start Feature Engineering===========")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
        use_turbulence=True,
        user_defined_feature=False,
    )


    processed = fe.preprocess_data(df)

    # Training & Trading data split
    train = data_split(processed, config.START_DATE, config.START_TRADE_DATE)
    trade = data_split(processed, config.START_TRADE_DATE, config.END_DATE)
    train.to_csv('/result_train',index=False)
    trade.to_csv('/result_trade',index=False)