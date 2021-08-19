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

def train_one(m_path,t_path):
    import dill as pickle
    f=open(m_path,'rb')
    trained_sac=pickle.load(f)
    f.close()
    f=open(t_path,'rb')
    e_trade_gym=pickle.load(f)
    f.close()
    #########################################################################
    print("==============Start Trading===========")
    '''
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=trained_sac, test_data=trade, test_env=env_trade, test_obs=obs_trade
    )
    '''
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        #model=all_model, environment=e_trade_gym
        model=trained_sac, environment=e_trade_gym
    )
    
    df_account_value.to_csv(
        "/df_account_value"
    )
    df_actions.to_csv("./" + config.RESULTS_DIR + "/df_actions_" + now + ".csv")