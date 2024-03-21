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
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import glob
from data_prepper import get_buoy_training_data
from matplotlib import pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import SGD
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from utils import calculate_mape




time_column_name = 'datetime'
directory_path = '/Users/jacobvanalmelo/code/deep_stocks/data'

def load_and_combine_jsons(directory_path, time_column_name):
    """
    Reads all JSON files from the provided directory, combines them into a single DataFrame,
    and sorts the combined DataFrame based on the specified time column.

    Parameters:
    - directory_path: str, the path to the directory containing the JSON files.
    - time_column_name: str, the name of the column used for sorting the data in time order.

    Returns:
    - combined_df: pandas.DataFrame, the combined DataFrame sorted based on the time column.
    """

    # Use glob to find all JSON files in the directory
    json_files = glob.glob(f"{directory_path}/*.json")

    # Initialize an empty list to store DataFrames
    dfs = []

    # Iterate over the JSON files, load each into a DataFrame, and append it to the list
    for file_path in json_files:
        df = pd.read_json(file_path)
        dfs.append(df)

    # Concatenate all DataFrames in the list into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    # Sort the combined DataFrame based on the time column
    combined_df.sort_values(by=time_column_name, inplace=True)

    return combined_df


# Example usage
# directory_path = 'path/to/your/directory'
# time_column_name = 'your_time_column'
# combined_df = load_and_combine_jsons(directory_path, time_column_name)
# print(combined_df.head())


def get_data_contiguous(dfX, dfy, n_timesteps, lead_time, num_outputs, timestep_size=3600):
    """
    Extracts contiguous blocks of data from input and target dataframes and prepares them for machine learning modeling (original case is LSTM).
    This function walks through the dataset and makes one sample (X and y pair) at a time, and if the timesteps within it are contiguous, it is added to the data set.


    Args:
        dfX (pandas.DataFrame): The input dataframe containing the features. It should have a 'time' column and
            other feature columns.
        dfy (pandas.DataFrame): The target dataframe containing the output values. It should have a 'time' column
            and other target columns.
        n_timesteps (int): The number of consecutive timesteps to include in each input sample
        lead_time (int): The number of timesteps between the last known value of X and the corresponding target value of y
        num_outputs (int): The number of consecutive timesteps to include in each target sample.
        timestep_size (int, optional): The duration of each timestep in seconds. Defaults to 3600 (seconds).

    Returns:
        tuple: A tuple containing three elements:
            - X (numpy.ndarray): A 3-dimensional array representing the input samples, with shape
              (num_samples, n_timesteps, num_features).
            - y (numpy.ndarray): A 3-dimensional array representing the target samples, with shape
              (num_samples, num_outputs, num_targets).
            - y_index (list): A list containing the timestamps corresponding to each target sample.

    Raises:
        ValueError: If the lengths of the input and target dataframes are not the same.
        KeyError: If the 'time' column does not exist in the input dataframe.

    Notes:
        - This function assumes that the input and target dataframes have consistent timestamps and order.
        - Any missing or irregular timesteps in the input data will result in the corresponding samples being skipped.
        - Any nan or None values in the input or target data will be replaced with 0.
    """
    # Error checking
    if len(dfX) != len(dfy):
        raise ValueError("Input and target dataframes must have the same length.")

    if 'time' not in dfX.columns:
        raise KeyError("'time' column must exist in the input dataframe.")

    # Extract list of features, excluding "time"
    X_features = [col for col in dfX.columns if col != 'time']

    # If there's a 'time' column in dfy, use it for y_index but exclude from y
    if 'time' in dfy.columns:
        y_features = [col for col in dfy.columns if col != 'time']
    else:
        y_features = dfy.columns.tolist()

    X, y, y_index = [], [], []
    max_index = len(dfX) - n_timesteps - lead_time - num_outputs + 1

    for i in tqdm(range(max_index), desc="Progress:", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
        total_block = dfX.iloc[i : i + n_timesteps + lead_time + num_outputs]

        time_diffs = total_block['time'].diff()[1:].dt.total_seconds()
        # if not all(time_diffs == timestep_size):
        #     continue

        X_block = dfX.iloc[i : i + n_timesteps][X_features]
        X.append(X_block.values)

        y_start = i + n_timesteps + lead_time
        y_end = y_start + num_outputs
        y_block = dfy.iloc[y_start : y_end][y_features]
        y.append(y_block.values.flatten().tolist())

        if 'time' in dfy.columns:
            y_index.extend(dfy.iloc[y_start : y_end]['time'].tolist())
        else:
            y_index.extend(dfX.iloc[y_start : y_end]['time'].tolist())

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    # Expanding dimensions of y to match with expected output
    if len(y.shape) == 1:
        y = np.expand_dims(y, axis=-1)

    # Replacing any nan or None values with 0
    X = np.nan_to_num(X, 0)
    y = np.nan_to_num(y, 0)

    return X, y, y_index
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





# todo scan folder and pull out .jsons
# for each run this function and append to a list


df = load_and_combine_jsons(directory_path, time_column_name)

def get_data(df):


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
    df_train_time = pd.DataFrame(df_train['time'])
    df_train_time.reset_index(inplace=True)
    df_train_time.drop('index', axis=1,inplace=True)

    # separate categorical and numerical
    # df_cat = df.select_dtypes(include = 'object')
    df_num = df_train.select_dtypes(exclude = 'object').drop(df_train_time.columns, axis=1)
    num_columns = df_num.columns

    # encode dummies, reset index so I can merge without nan
    # dummy_df_cat = pd.get_dummies(df_cat)
    # dummy_df_cat.reset_index(inplace=True)
    # dummy_df_cat.drop('index', axis=1,inplace=True)
    # scale numberical columns
    # if scaler == None:
    Xscaler.fit(df_num)
    dfX_train_processed = pd.DataFrame(Xscaler.transform(df_num), columns=num_columns)
    dfX_train_processed = pd.concat([dfX_train_processed, df_train_time], axis = 1)
    #     return processed_df, Xscaler
    # else:
    # dfX_train_processed, Xscaler= buoy_encodeNscaleX(dfX_train)


    joblib.dump(Xscaler, "Xscaler_stock.pkl")
    joblib.dump(yscaler, "yscaler_stock.pkl")
    # separate test X and y
    dfy_test = pd.DataFrame(df_test[target].copy())


    scaler = Xscaler

    #   separate time, reset index so I can merge without nan
    df_test_time = pd.DataFrame(df_test['time'])
    df_test_time.reset_index(inplace=True)
    df_test_time.drop('index', axis=1,inplace=True)


    # separate categorical and numerical
    # df_cat = df.select_dtypes(include = 'object')
    df_num = df_test.select_dtypes(exclude = 'object').drop(df_test_time.columns, axis=1)
    num_columns = df_num.columns

    # encode dummies, reset index so I can merge without nan
    # dummy_df_cat = pd.get_dummies(df_cat)
    # dummy_df_cat.reset_index(inplace=True)
    # dummy_df_cat.drop('index', axis=1,inplace=True)
    # scale numberical columns
    # if scaler == None:
    #     Xscaler.fit(df_num)
    #     df_num_scaled = pd.DataFrame(Xscaler.transform(df_num), columns=num_columns)
    dfX_test_processed = pd.DataFrame(scaler.transform(df_num), columns=num_columns)
    dfX_test_processed = pd.concat([dfX_test_processed, df_test_time], axis = 1)
    #     return processed_df, Xscaler
    # else:


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




    X_train, y_train, traindex = get_data_contiguous(dfX=zdf_train, dfy=dfy_train_processed, n_timesteps=n_timesteps, num_outputs=n_outputs, lead_time=lead_time, timestep_size=120)
    X_test, y_test, testdex = get_data_contiguous(dfX=zdf_test, dfy=dfy_test_processed, n_timesteps=n_timesteps, num_outputs=n_outputs, lead_time=lead_time, timestep_size=120)


return all_that_shit #I mean ALLL that shiiitt....








n_timesteps = X_train.shape[1]
n_features = X_train.shape[2]

# # define model
# epochs= 10
# batch_size = 32
# verbose = 1
# lrate=.004
# loss_func = 'huber'
# momentum = .9
# input_nodes = 8
# # opt = SGD(learning_rate=lrate,  decay = 1e-6, momentum=momentum, nesterov=True)
# opt = SGD(learning_rate=lrate, momentum=momentum)

# # opt = SGD(learning_rate=lrate,  momentum=momentum)
# # define model
# def make_model(initial_lr=0.004, decay_rate=0.0001, decay_steps=100):
#     model = Sequential()
#     model.add(LSTM(input_nodes, activation='relu', return_sequences=False, kernel_initializer='he_uniform', input_shape=(n_timesteps, n_features)))
#     model.add(Dense(2))
#     model.add(Dense(1))

#     # Define the learning rate schedule
#     lr_schedule = ExponentialDecay(
#         initial_learning_rate=initial_lr,
#         decay_steps=decay_steps,
#         decay_rate=decay_rate,
#         staircase=True)  # 'staircase=False' for smooth decay, 'True' for discrete steps

#     # Incorporate the learning rate schedule into the optimizer
#     optimizer = Adam(learning_rate=lr_schedule)

#     model.compile(loss=loss_func, optimizer=optimizer)
#     return model
# model = make_model()

# history = model.fit(X_train, y_train, epochs=epochs, verbose=verbose, validation_data=(X_test, y_test))
# loss = history.history['loss']
# val_loss = history.history.get('val_loss', None)  # This might not exist if validation was not performed

# # Creating the plot for the training loss
# plt.plot(loss, label='Training Loss')

# # If there's validation loss, add it to the plot
# if val_loss is not None:
#     plt.plot(val_loss, label='Validation Loss')

# # Adding titles and labels
# plt.title('Model Loss Over Epochs')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend()

# # Display the plot
# plt.show()






# yhat = model.predict(X_test)
# yhat_unscaled = yscaler.inverse_transform(yhat)
# y_test_unscaled = yscaler.inverse_transform(y_test)
# mape = calculate_mape(yhat, y_test)
# print(f'\n\n\n\nACTUAL MAPE:{mape}\n')
# ###############################################################3 BELOW IS UNVERIFIED

# pio.renderers.default='browser'

# # Create traces
# fig = go.Figure()
# fig.add_trace(go.Scatter(y=yhat_unscaled[:,0],
#                     mode='lines',
#                     name='yhat_unscaled'))
# fig.add_trace(go.Scatter(y=y_test_unscaled[:,0],
#                     mode='lines',
#                     name='y_test_unscaled'))
# # fig.add_trace(go.Scatter(y=y_test_unscaled[:,0],
# #                     mode='lines',
# #                     name='y_test_unscaled'))
# fig.update_layout(
#     title="Actual vs Predicted Condition of lanai buoy (51213) based on '51003', '51004', '51211', and '51212', using 0 hour lead and the previous 16 hours back"
# )

# fig.show()

# define model
from keras.layers import Dropout

epochs= 20
# epochs= 15
batch_size = 32
verbose = 1
lrate=.01
momentum = .9
# decay = 0.0005
# decay = .0005

opt = SGD(learning_rate=lrate,  momentum=momentum)
print(n_timesteps, n_features, n_outputs, epochs, batch_size)
# define model
model = Sequential()
model.add(LSTM(7, activation='relu', return_sequences=False,kernel_initializer='he_uniform', input_shape=(n_timesteps, n_features)))
model.add(Dropout(.1))
# model.add(LSTM(20, activation='relu'))
# model.add(Dropout(.1))
model.add(Dense(2))
model.add(Dense(1))
model.compile(loss='mean_absolute_error', optimizer=opt)
history = model.fit(X_train, y_train, epochs=epochs, verbose=verbose)
new_history = model.fit(X_train, y_train, epochs=50, verbose=verbose)
filename = 'buoy_model_202306010.sav'
joblib.dump(model, filename)

loss = history.history.get("loss")
loss = new_history.history.get("loss")
plt.plot(loss)
plt.title("Loss History")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

"""OK....

for a first run. now lets get yhat and see how it compares to y_test
"""
def mape(yhat, y_test):
    n = len(yhat)
    mape = 100 * (1/n) * np.sum(np.abs((y_test-yhat)/y_test))
    return mape
def mape(yhat, y_test):
    n = len(yhat)
    mape_sum = 0
    for i in range(n):
        mape_sum += np.abs((y_test[i]-yhat[i])/y_test[i])
    mape = 100 * (1/n) * mape_sum
    return mape
def mape(yhat, y_test):
  errors = np.abs((yhat - y_test)/y_test)
  mape = 100 * errors.mean()
  return mape

yhat = model.predict(X_test)
'''
whoops, I'll have to fit a scaler directly to y in the future I guess... this is imperfect...
'''
yhat_unscaled = yscaler.inverse_transform(yhat)
y_test_unscaled = yscaler.inverse_transform(y_test)
mape = mape(yhat, y_test)
###############################################################3 BELOW IS UNVERIFIED

pio.renderers.default='browser'

# Create traces
fig = go.Figure()
fig.add_trace(go.Scatter(y=yhat_unscaled[:,0],
                    mode='lines',
                    name='yhat_unscaled'))
fig.add_trace(go.Scatter(y=y_test_unscaled[:,0],
                    mode='lines',
                    name='y_test_unscaled'))
# fig.add_trace(go.Scatter(y=y_test_unscaled[:,0],
#                     mode='lines',
#                     name='y_test_unscaled'))
fig.update_layout(
    title="Actual vs Predicted Condition of 51205 based on 51000 and 51101 using 12 hour lead"
)

fig.show()



# include array indices (timestamps)
# why is it off by 57? predictions are close byt off (in the graph)

def up_or_down(df, target_column, lookback_windows, threshold):
    """
    Adds multiple columns to the input DataFrame indicating the direction of change in the target column's value.
    Each new column corresponds to a lookback window specified in 'lookback_windows'. The function determines if the value in
    the target column has gone up, down, or remained unchanged within a threshold over each lookback period.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - target_column (str): The name of the column to analyze for value changes.
    - lookback_windows (list of int): A list of integers where each integer specifies a lookback window length.
    - threshold (float): The minimum change required to consider the value as having moved up or down.

    Returns:
    - pd.DataFrame: The original DataFrame with added columns for each lookback window indicating the direction of change.

    The added columns are named 'direction_{lookback_window}-out', where {lookback_window} is replaced with the actual window size.
    Each cell in these columns can have a value of 1 (indicating an increase), -1 (indicating a decrease), or 0 (indicating no significant change).
    """

    # Validate inputs
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame.")
    if target_column not in df.columns:
        raise ValueError(f"{target_column} is not a column in the DataFrame.")
    if not all(isinstance(window, int) and window > 0 for window in lookback_windows):
        raise ValueError("lookback_windows must be a list of positive integers.")
    if not isinstance(threshold, float) and not isinstance(threshold, int):
        raise ValueError("threshold must be a float or an int.")

    # Iterate over each lookback window to create new columns
    for window in lookback_windows:
        # Compute the difference between the current value and the value 'window' rows back
        df[f'direction_{window}out'] = df[target_column].diff(periods=window)

        # Apply the threshold to determine direction
        df[f'direction_{window}out'] = np.select(
            [
                df[f'direction_{window}out'] > threshold,
                df[f'direction_{window}out'] < -threshold
            ],
            [
                1,  # Value went up
                -1  # Value went down
            ],
            default=0  # No significant change
        )

    return df

# Example usage:
df = pd.DataFrame({'Price': [100, 105, 103, 108, 110, 109, 111]})
lookback_windows = [1, 2, 3]
threshold = 1.5
df_with_direction = up_or_down(df, 'Price', lookback_windows, threshold)
print(df_with_direction)

# Here are notes for a new function to assess whether the price will break certain bounds within the next window

def BounderBreakers():
    '''
    this function takes in a dataframe and for each timestamp looks backward (back_window) and creates a new colum "future_movement that indicates
    how the stock value moves in the future by computing the mean and stdv for all of the previous low values and the same for the high values, 
    then looks forwrd forward_window timesteps and determines whether the forward highs break the [previous high stdv *2] first (1) 
    or whther the anologue occurs with the low values first (-1) if neither (0) or both (NaN) occur within the forward window.


    ARGUMENTS:
    boundary_style: is 
        back_window: how many timesteps backward are we looking? (60)
        front_winow: how many timesteps foward are we looking? (10)
        df: pandas dataframe input that has 'high', 'low', and 'close' attributes

    This function takes in a pandas dataframe and creates a new column that has the value future_movement and denotes 
    whether the price exceeds the upper (1) the lower(-1) or neither (0) or both (-1) within the front_window time ahead

    OUTPUT: df with a new column labeled "future_movement"
    
    '''


    return df



















