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
    trade = pd.read_csv(t_path)
    #########################################################################
    print("==============Start Trading===========")
    '''
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=trained_sac, test_data=trade, test_env=env_trade, test_obs=obs_trade
    )
    '''
    env_kwargs = {
        "hmax": 100, 
        "initial_amount": 1000000, 
        "buy_cost_pct": 0.001, 
        "sell_cost_pct": 0.001, 
        "state_space": state_space, 
        "stock_dim": stock_dimension, 
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST, 
        "action_space": stock_dimension, 
        "reward_scaling": 1e-4
        }
    e_trade_gym = StockTradingEnv(df=trade, turbulence_threshold=250.0, **env_kwargs)
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        #model=all_model, environment=e_trade_gym
        model=trained_sac, environment=e_trade_gym
    )
    
    df_account_value.to_csv(
        "/df_account_value"
    )
    df_actions.to_csv("./" + config.RESULTS_DIR + "/df_actions_" + now + ".csv")