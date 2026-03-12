import yfinance as yf
from pypots.representation import TS2Vec
import numpy as np
import pandas as pd
from pathlib import Path
import os
import yaml
from typing import Optional, Literal
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # maybe there is a better way?
#TO IMPROVE SEEDING 

class ts2vec():

    def __init__(self,
                 data:pd.DataFrame,
                 scaler:Literal['Standard','MinMaxScaler','Returns','LogReturns'] = 'LogReturns'):

        
        self.model_path = ROOT_DIR + '/trained_models/'

        self.features = sorted(data.columns.get_level_values(0).unique())
        self.tickers  = sorted(data.columns.get_level_values(1).unique())

        self.processed_data = self._preprocess(data,scaler)

        self.n_samples = len(self.tickers)
        self.n_features = len(self.features)
        self.n_steps = self.processed_data.shape[1]

        self.model = None # preset model as none

        self.DEFAULT_SETTINGS_TS2VEC:dict = {
            'n_features':self.n_features,
            'n_steps':self.n_steps,
            'n_output_dims':64,
            'd_hidden':64,
            'n_layers':3,
            'mask_mode':"binomial",
            'batch_size':32,
            'epochs':30,
            'model_saving_strategy':"best",
            'verbose':True,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        }

    def _preprocess(self, data, scaler):

        if not isinstance(data, pd.DataFrame):
            raise TypeError("input must be a Pandas DataFrame")

        if not isinstance(data.index,pd.DatetimeIndex):
            raise TypeError("data index must be a Pandas DatetimeIndex")
        
        if not isinstance(data.columns,pd.MultiIndex):
            raise TypeError("data columns must be a Pandas MultiIndex")

        match scaler:
            case 'Standard':
                prep = self._Standard(data)
            case 'MinMaxScaler':
                prep = self._minmaxScaler(data)
            case 'LogReturns':
                prep = self._logRet(data)

        return self._2tensor(prep)
        

    def _Standard(self, data):
        scaler = StandardScaler()
        arr = scaler.fit_transform(data.values)           
        df = pd.DataFrame(arr, index=data.index, columns=data.columns)
        return df.dropna()
    

    def _minmaxScaler(self,data):
        scaler = MinMaxScaler()
        arr = scaler.fit_transform(data.values)           
        df = pd.DataFrame(arr, index=data.index, columns=data.columns)
        return df.dropna()
    

    def _logRet(self, data):
        return np.log(data / data.shift(1)).dropna()
    

    def _2tensor(self,data):

        x = []
        for tic in self.tickers:
            df_t = data.xs(tic, level=1, axis=1)     
            df_t = df_t[self.features]        

            x.append(df_t.values)              

        tensor = np.stack(x, axis=0)           

        print("tensor shape:", tensor.shape)

        return tensor


    def train_model(self,settings:Optional[dict|Path|str] = None, model_name:str = None):

        self.settings = self.DEFAULT_SETTINGS_TS2VEC.copy()
        usr_settings = {}
        
        if isinstance(settings,(Path,str)):
            'reading yml or yaml file'
            with open(settings, 'r') as file:
                usr_settings = yaml.safe_load(file)
        elif isinstance(settings, dict):
            usr_settings = settings

        self.settings.update(usr_settings)

        self.model = TS2Vec(
            **self.settings,
          )
        
        self.model.fit({'X': self.processed_data,
                        'y':np.arange(self.n_samples)
                        })
        if model_name is not None:
            self.model.save(saving_path=self.model_path + model_name, overwrite=True)
            self._last_model_name = model_name 


    def ts_embeddings(self,
                      load_model:Optional[bool|str]=True,
                      data:Optional[pd.DataFrame] = None, #portion of data which u want the embeddings
                      scaler:Optional[Literal['Standard','MinMaxScaler','Returns','LogReturns']] = 'LogReturns',
                      sliding_padding:int = 0):
        
        if self.model is None and load_model is False:
            raise RuntimeError("No model in memory. Either train first or pass load_model path.")

        if load_model is True:
            # use whatever was last saved — requires model_name to be stored
            if not hasattr(self, '_last_model_name'):
                raise RuntimeError("No model name stored. Pass an explicit path string.")
            path = os.path.join(self.model_path, f"{self._last_model_name}.pypots")
            self.model.load(path)

        elif isinstance(load_model, str):
            path = os.path.join(self.model_path, f"{load_model}.pypots")
            self.model = TS2Vec(**self.DEFAULT_SETTINGS_TS2VEC)
            self.model.load(path)

        to_encode = self._preprocess(data, scaler) if data is not None else self.processed_data

        embeddings = self.model.represent(
            {'X': to_encode},
            sliding_padding=sliding_padding
        )
        # shape: (n_tickers, n_steps, n_output_dims)
        return embeddings