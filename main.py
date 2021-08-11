import json
import logging
import os
import time
from argparse import ArgumentParser
import datetime
import sys
from finrl.config import config


def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        dest="mode",
        help="start mode, train, download_data" " backtest",
        metavar="MODE",
        default="train",
    )
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()
    if not os.path.exists("./" + config.DATA_SAVE_DIR):
        os.makedirs("./" + config.DATA_SAVE_DIR)
    if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
        os.makedirs("./" + config.TRAINED_MODEL_DIR)
    if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
        os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
    if not os.path.exists("./" + config.RESULTS_DIR):
        os.makedirs("./" + config.RESULTS_DIR)

    if options.mode == "train":
        if(sys.argv[1]=='fetch'):
            import finrl.autotrain.fetch
            finrl.autotrain.fetch.train_one()
        elif(sys.argv[1]=='feature'):
            import finrl.autotrain.feature
            finrl.autotrain.feature.train_one(sys.argv[2])
        elif(sys.argv[1]=='train_model'):
            import finrl.autotrain.train_model
            finrl.autotrain.train_model.train_one(sys.argv[2],sys.argv[3])
        # import finrl.autotrain.training

        # finrl.autotrain.training.train_one()

    elif options.mode == "download_data":
        from finrl.marketdata.yahoodownloader import YahooDownloader
        print('开始下载数据……')
        df = YahooDownloader(start_date=config.START_DATE,
                             end_date=config.END_DATE,
                             ticker_list=config.DOW_30_TICKER).fetch_data()
        now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")
        df.to_csv("./" + config.DATA_SAVE_DIR + "/" + now + ".csv")

        
if __name__ == "__main__":
    main()
