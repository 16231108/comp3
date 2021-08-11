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

def train_one(path1,path2):
    train = pd.read_csv(path1)
    trade = pd.read_csv(path2)
    #print('trade is:',trade)
    # calculate state action space
    stock_dimension = len(train.tic.unique())
    state_space = (
        1
        + 2 * stock_dimension
        + len(config.TECHNICAL_INDICATORS_LIST) * stock_dimension
    )

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

    e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    e_trade_gym = StockTradingEnv(df=trade, turbulence_threshold=250.0, **env_kwargs)
    lxc_trade_gym =lxcStockTradingEnv(df=trade, turbulence_threshold=250.0, **env_kwargs)
    e_trade_gym2 = StockTradingEnv(df=trade, turbulence_threshold=250.0, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    lxc_env_train, _ = lxc_trade_gym.get_sb_env()
    env_trade, obs_trade = e_trade_gym.get_sb_env()
    agent = DRLAgent(env=env_train)
    lxc_agent = DRLAgent(env=lxc_env_train)

    print("==============Model Training===========")
    all_model = []
    print("start training ddpg model")
    now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")


    ###################################################################################################a2c
    print("start training a2c model")
    model_a2c = agent.get_model("a2c")
    trained_a2c = agent.train_lxc_model(
        model=model_a2c, tb_log_name="a2c", total_timesteps=4000,lxcType=None,lxcName="lxc2"
    )
    #print('trained_a2c is:', trained_a2c)
    all_model.append(trained_a2c)


    ####################################################################sac


    print("start training sac model")
    model_sac = agent.get_model("sac")
    trained_sac = agent.train_lxc_model(
        model=model_sac, tb_log_name="sac", total_timesteps=4000, lxcType=None, lxcName="lxc1"
    )    
    #print('trained_sac is:', trained_sac)
    all_model.append(trained_sac)
    #################################################################ddpg

    model_ddpg = agent.get_model("ddpg")
    trained_ddpg = agent.train_lxc_model(
        model=model_ddpg, tb_log_name="ddpg", total_timesteps=4000,lxcType=None,lxcName="lxc1"
    )
    #print('trained_ddpg is:',trained_ddpg)
    all_model.append(trained_ddpg)