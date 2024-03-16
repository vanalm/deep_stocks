#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 18:10:30 2024

@author: jacobvanalmelo
"""
from utils import smooth_dataframe, df_traintest, encode_dummies, get_data_contiguous, load_scalers, calculate_mape, buoy_encodeNscaleX, buoy_scaley, buoy_setcols
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd



# Function to scale y
def buoy_scaley(dfy, default_scaler=MinMaxScaler(), scaler=None):
    """
    Scale a single numerical column, target dataframe, either with a prefit scaler provided, or by fitting and returning one

    dfy: a single numerical column dataframe
    scaler: an optional scaling function, defaulting to sklearn's StandardScaler

    Returns a scaled dataframe, as well as the scaling function if one is not provided
    """
    # check if parameter is a DataFrame
    if not isinstance(dfy, pd.DataFrame):
        raise TypeError("Error: The dfy argument must be a Pandas DataFrame object")
    try:
        # scale numberical columns
        y_column = dfy.columns
        if scaler == None:
            yscaler = default_scaler
            yscaler.fit(dfy)
            dfy_scaled = pd.DataFrame(yscaler.transform(dfy), columns=y_column)
            return dfy_scaled, yscaler
        else:
            dfy_scaled = pd.DataFrame(scaler.transform(dfy), columns=y_column)
            return dfy_scaled

    except Exception as e:
        print(f'An error occurred while attempting to scale the input dataframe: {e}')




filename = "~/downloads/Fri-Dec-01-2023-09_30_00-GMT-0500.json"

# todo scan folder and pull out .jsons
# for each run this function and append to a list



def get_data(filename):
    df = pd.read_json(filename)
    
    
    columns_to_convert = ['askVolume', 'bidVolume']
    
    # Convert the specified columns to floats
    # df[columns_to_convert] = df[columns_to_convert].astype(float)
    
    for column in columns_to_convert:
        df[column] = df[column].apply(lambda d: float(d.get("low", 0)) if isinstance(d, dict) else 0)
    
    df['time'] = df['datetime'].copy()
    df.drop('datetime', axis=1, inplace=True)
    target = 'high'
    
    df_train, df_test = df_traintest(df, .2)
    print(f'train size = {len(df_train)}\ntest size = {len(df_test)}')
    print(f'df_train.info():\n{df.info()}')
    
    dfy_train = pd.DataFrame(df_train[target].copy())
    # scale ytrain, and retrieve scaler
    dfy_train_processed,  yscaler = buoy_scaley(dfy_train)
    
    
    # define X
    dfX_train = df_train.drop(target, axis=1)
    # encode and scale X, retrieve scaler
    print(f' looking for "time" in this... and why do we need time in it and where did we drop it? {dfX_train.info()}')
          
    Xscaler=MinMaxScaler(feature_range=(0,1))
    #   separate time, reset index so I can merge without nan
    df_time = pd.DataFrame(df_train['time'])
    df_time.reset_index(inplace=True)
    df_time.drop('index', axis=1,inplace=True)
    
    # separate categorical and numerical
    # df_cat = df.select_dtypes(include = 'object') 
    df_num = df_train.select_dtypes(exclude = 'object').drop(df_time.columns, axis=1)
    num_columns = df_num.columns
    
    # encode dummies, reset index so I can merge without nan
    # dummy_df_cat = pd.get_dummies(df_cat)
    # dummy_df_cat.reset_index(inplace=True)
    # dummy_df_cat.drop('index', axis=1,inplace=True)
    # scale numberical columns
    # if scaler == None:
    Xscaler.fit(df_num)
    dfX_train_processed = pd.DataFrame(Xscaler.transform(df_num), columns=num_columns)
    # processed_df = pd.concat([df_num_scaled, dummy_df_cat, df_time], axis = 1)
    #     return processed_df, Xscaler
    # else:
    # dfX_train_processed, Xscaler= buoy_encodeNscaleX(dfX_train)
    
    
    joblib.dump(Xscaler, "Xscaler_stock.pkl")
    joblib.dump(yscaler, "yscaler_stock.pkl")
    # separate test X and y
    dfy_test = pd.DataFrame(df_test[target].copy())
    
    
    scaler = Xscaler
    
    #   separate time, reset index so I can merge without nan
    df_time = pd.DataFrame(df_test['time'])
    df_time.reset_index(inplace=True)
    df_time.drop('index', axis=1,inplace=True)
        
    
    # separate categorical and numerical
    # df_cat = df.select_dtypes(include = 'object') 
    df_num = df_test.select_dtypes(exclude = 'object').drop(df_time.columns, axis=1)
    num_columns = df_num.columns
    
    # encode dummies, reset index so I can merge without nan
    # dummy_df_cat = pd.get_dummies(df_cat)
    # dummy_df_cat.reset_index(inplace=True)
    # dummy_df_cat.drop('index', axis=1,inplace=True)
    # scale numberical columns
    # if scaler == None:
    #     Xscaler.fit(df_num)
    #     df_num_scaled = pd.DataFrame(Xscaler.transform(df_num), columns=num_columns)
    #     processed_df = pd.concat([df_num_scaled, dummy_df_cat, df_time], axis = 1)
    #     return processed_df, Xscaler
    # else:
    dfX_test_processed = pd.DataFrame(scaler.transform(df_num), columns=num_columns)
    
    
        # processed_df = pd.concat([df_num_scaled, dummy_df_cat, df_time], axis = 1)
        # return processed_df
    
    
    
    
    
    # apply prefit Xscaler and yscaler to test
    # dfX_test_processed = buoy_encodeNscaleX(dfX_test, scaler =Xscaler )
    dfy_test_processed = buoy_scaley(dfy_test, scaler=yscaler)
    dfX_test_processed.shape[1] == dfy_test_processed.shape[1]
    print('former:', dfX_train_processed.shape)
    print('latter', dfX_test_processed.shape)
    # make sure the columns in each test and train are consistent
    dfX_train_processed_same, dfX_test_processed_same = buoy_setcols(dfX_train_processed, dfX_test_processed)
    dfX_train_processed_same.shape[1] ==dfX_test_processed_same.shape[1]
    dfX_train_processed_same.columns==dfX_test_processed_same.columns
    
    
    n_outputs = 1
    n_timesteps = 20
    lead_time = 0 #
    
    zdf_train = dfX_train_processed.copy()
    zdf_test = dfX_test_processed.copy()
    
    
    zdf_train['time'] = pd.to_datetime(zdf_train['time'])
    zdf_test['time'] = pd.to_datetime(zdf_test['time'])
    
    
    
    
    X_train, y_train, traindex = get_data_contiguous(dfX=zdf_train, dfy=dfy_train_processed, n_timesteps=n_timesteps, num_outputs=n_outputs, lead_time=lead_time)
    X_test, y_test, testdex = get_data_contiguous(dfX=zdf_test, dfy=dfy_test_processed, n_timesteps=n_timesteps, num_outputs=n_outputs, lead_time=lead_time)
    
    
    return data






































