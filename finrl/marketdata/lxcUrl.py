# -*- coding: utf-8 -*-
import requests
import datetime
import pandas as _pd
import base64
import time
from tqdm import tqdm
import json
app_key = "81118a71-6e2d-4117-a03e-71c1e405faef"
app_secrect = "26b193b3-e8da-4eed-a613-147388f17acd"
token = 'F536FCFA8EF34D2D923060768394E3212021030809454981118A71'

def getEveryDay(begin_date,end_date):
    # 前闭后闭
    date_list = []
    begin_date = datetime.datetime.strptime(begin_date, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date,"%Y-%m-%d")
    while begin_date <= end_date:
        date_str = begin_date.strftime("%Y-%m-%d")
        date_list.append(date_str)
        begin_date += datetime.timedelta(days=1)
    return date_list
def getToken(app_key,app_secrect):
	global token
	bytesString = (app_key+':'+app_secrect).encode(encoding="utf-8")
	url = 'https://sandbox.hscloud.cn/oauth2/oauth2/token';
	header = {'Content-Type': 'application/x-www-form-urlencoded',
		'Authorization': 'Basic '+str(base64.b64encode(bytesString),encoding="utf-8")}
	field = {'grant_type' : 'client_credentials'}
	r = requests.post(url,data=field,headers=header)
	if r.json().get('access_token') :
		token = r.json().get('access_token')
		print("获取公共令牌:"+str(token))
		return
	else :
		print("获取公共令牌失败")
		exit
def postOpenApi(url,params):
    global token
    header = {'Content-Type': 'application/x-www-form-urlencoded',
		'Authorization': 'Bearer '+token}
    r = requests.post(url,data=params,headers=header)
    temp = r.json().get('data')
    #print(temp[0]['high_price']=="")
    #print("result = "+str(r.json().get('data')))
    return temp
def hsDownloadData(en_prod_code,begin_date,end_date):
    dataList = getEveryDay(begin_date,end_date)
    url = "https://sandbox.hscloud.cn/gildataastock/v1/astock/quotes/daily_quote"
    Date =[]
    Open = []
    High = []
    Low = []
    Close = []
    Adj_Close = []
    Volume = []
    for oneDay in tqdm(dataList):
        #params = 'en_prod_code=600000.SH&trading_date=2016-12-30&unit=0'
        params = "en_prod_code="+en_prod_code+"&trading_date="+oneDay
        #print(params)
        temp = postOpenApi(url, params)
        if(temp[0]['high_price'] != ""):#有数据，开盘
            Date.append(temp[0]['trading_date'])
            Open.append(temp[0]['open_price'])
            High.append(temp[0]['high_price'])
            Low.append(temp[0]['low_price'])
            Close.append(temp[0]['close_price'])#后期需要修改
            Adj_Close.append(temp[0]['avg_price'])
            Volume.append(temp[0]['business_amount'])
        time.sleep(2)
    Frame = {"Open": Open,
             "High": High,
             "Low": Low,
             "Close": Close,
             "Adj Close": Adj_Close,
             "Volume": Volume

    }
    quotes =_pd.DataFrame.from_dict(Frame)
    quotes.index = _pd.to_datetime(Date)
    quotes.sort_index(inplace=True)
    quotes.index.name = "Date"
    return quotes
def jsonToDate(jsonDate):
    pass
if __name__ == '__main__':
    # getToken(app_key,app_secrect)
    data = hsDownloadData('000002.SZ','2020-12-21','2021-01-01')
    print(len(data))
    data = hsDownloadData('000001.SZ', '2020-12-21', '2021-01-01')
    print(len(data))
    print(type(data))
