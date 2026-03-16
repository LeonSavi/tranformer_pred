"""

Reference: https://github.com/AI4Finance-Foundation/FinRL_Crypto/blob/master/processor_Binance.py
CORRELATION THRESHOLD SET TO 0.9 LIKE IN THE PAPER
"""

import pandas as pd
from datetime import datetime
import numpy as np
import os
from binance.client import Client
from talib import RSI, MACD, CCI, DX, ROC, ULTOSC, WILLR, OBV, HT_DCPHASE
from icecream import ic
import datetime as dt
from utils.parser import Colours, YMLparser
import pickle
from utils.API import API_KEY_BINANCE, API_SECRET_BINANCE

from datetime import datetime

class BinanceData():

    def __init__(self):
        self.end_date = None
        self.start_date = None
        # self.tech_indicator_list = None
        self.correlation_threshold = None
        self.binance_api_key = API_KEY_BINANCE  # Enter your own API-key here
        self.binance_api_secret = API_SECRET_BINANCE  # Enter your own API-secret here
        self.binance_client = Client(api_key=API_KEY_BINANCE, api_secret=API_SECRET_BINANCE)
        self._foldercheck()


    def run(self,
            configs:str|dict,
            csv_format:bool = True):
        
        self.config_file = configs

        self._assign_config(configs)

        print('Downloading data from Binance...')
        data = self.download_data()
        print('Downloading finished! Transforming data...')
        data = self.clean_data(data)
        data = data.drop(columns=['time'])
        data['timestamp'] = self.servertime_to_datetime(data['timestamp'])
        data = data.set_index('timestamp')

        data = self.add_technical_indicator(data, self.tech_indicator_list)
        data = self.drop_correlated_features(data)

        if self.vix:
            data = self.add_vix(data)

        file_name = f"data/train_data_{self.time_interval}" 

        if csv_format:
            data.to_csv(f'{file_name}.csv')
        else:
            data.to_pickle(f'{file_name}.pkl')

        return data

    # main functions
    def download_data(self,):

        final_df = pd.DataFrame()

        if isinstance(self.ticker_list, str):
            ic(self.ticker_list)

        total = len(self.ticker_list)

        for num,i in enumerate(self.ticker_list):
            
            hist_data = self.get_binance_bars(
                self.start_date,
                self.end_date,
                self.time_interval,
                symbol=i)
            
            df = hist_data.iloc[:-1].copy()
            # df = df.dropna() # REMOVED CLEANING 
            df['tic'] = i
            final_df = pd.concat([final_df,df],ignore_index=True).reset_index(drop=True)

            print(f"DOWNLOADED {i} -- Completition Time: {round((num/total)*100)}% -- H: {datetime.now().strftime('%H:%M:%S')}"            )

        return final_df


    def clean_data(self, df):
        df = df.dropna()
        return df


    def add_technical_indicator(self, df, tech_indicator_list):
        final_df = pd.DataFrame()
        for i in df.tic.unique():
            # use massive function in previous cell
            coin_df = df[df.tic == i].copy()
            coin_df = self.get_TALib_features_for_each_coin(coin_df)

            # Append constructed tic_df
            final_df = pd.concat([final_df,coin_df])

        return final_df


    def drop_correlated_features(self, df):

        if not self.tech_indicator_list:
            corr_matrix = pd.DataFrame(df).corr(numeric_only=True).abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.correlation_threshold)]

            to_drop.remove('close')
            print(f'{Colours.CYAN}According to analysis, drop:{Colours.YELLOW} {to_drop}{Colours.END}')
            
            to_drop_for_consistency = ['open','high','low']
            to_drop.extend(to_drop_for_consistency)
            df_uncorrelated = df.drop(list(set(to_drop)), axis=1)

            # updatating
            self.tech_indicator_list = [i for i in df_uncorrelated.columns]
            self.config_training['tech_indicator_list'] = self.tech_indicator_list
            
            if isinstance(self.config_file,str):
                YMLparser(self.config_file).update_yml(
                    'training',{'tech_indicator_list':self.tech_indicator_list}
                    )

        else:
            print(f'{Colours.CYAN}Analysis already done, keeping:{Colours.YELLOW} {self.tech_indicator_list}{Colours.END}', )
            df_uncorrelated = df[self.tech_indicator_list]

        return df_uncorrelated


    # helper functions
    def stringify_dates(self, date: datetime):
        return str(int(date.timestamp() * 1000))


    def servertime_to_datetime(self, timestamp):
        list_regular_stamps = [0] * len(timestamp)
        for indx, ts in enumerate(timestamp):
            list_regular_stamps[indx] = dt.datetime.fromtimestamp(ts / 1000)
        return list_regular_stamps


    def get_binance_bars(self, start_date, end_date, kline_size, symbol):
        data_df = pd.DataFrame()

        klines = self.binance_client.get_historical_klines(
            symbol,
            kline_size,
            start_date,
            end_date)
        
        data = pd.DataFrame(klines,
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av',
                                     'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
        
        data = data.drop(labels=['close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'], axis=1)
        
        if len(data_df) > 0:
            temp_df = pd.DataFrame(data)
            data_df = data_df.append(temp_df)
        else:
            data_df = data

        data_df = data_df.apply(pd.to_numeric, errors='coerce')

        data_df['time'] = [datetime.fromtimestamp(x / 1000.0) for x in data_df.timestamp]
        # data.drop(labels=["timestamp"], axis=1)
        data_df.index = [x for x in range(len(data_df))]

        return data_df


    def get_TALib_features_for_each_coin(self, tic_df):

        tic_df['rsi'] = RSI(tic_df['close'], timeperiod=14)
        tic_df['macd'], _, _ = MACD(tic_df['close'], fastperiod=12,
                                    slowperiod=26, signalperiod=9)
        tic_df['cci'] = CCI(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
        tic_df['dx'] = DX(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
        tic_df['roc'] = ROC(tic_df['close'], timeperiod=10)
        tic_df['ultosc'] = ULTOSC(tic_df['high'], tic_df['low'], tic_df['close'])
        tic_df['willr'] = WILLR(tic_df['high'], tic_df['low'], tic_df['close'])
        tic_df['obv'] = OBV(tic_df['close'], tic_df['volume'])
        tic_df['ht_dcphase'] = HT_DCPHASE(tic_df['close'])

        return tic_df
    

    def _assign_config(self,configs:str|dict):

        if isinstance(configs,str) and (configs.endswith('.yml') or configs.endswith('.yaml')):
            self.config_training = YMLparser(configs).get('training')
        elif isinstance(configs,dict):
            self.config_training = configs
        else:
            ic(configs, type(configs))
            raise TypeError(f'Config must be either dict or a path to a yml/yaml')
        
        for key, value in self.config_training.items():
            setattr(self, key, value)


    def binance_to_multiindex(flat_df: pd.DataFrame) -> pd.DataFrame:
        """Convert BinanceData output to MultiIndex format for ts2vec wrapper."""
        flat_df = flat_df.reset_index()  # timestamp becomes column
        pivoted = flat_df.pivot(index='timestamp', columns='tic')
        # columns are now (feature, ticker) MultiIndex — exactly what ts2vec expects
        pivoted.index = pd.to_datetime(pivoted.index)
        return pivoted


    @staticmethod
    def _foldercheck():
        if not os.path.exists('data/'):
            print(f'{Colours.HEADER} Created folder "data"{Colours.END}')
            os.mkdir('data')