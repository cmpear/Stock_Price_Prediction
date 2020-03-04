from io import StringIO
import requests
import json
import pandas as pd
import types
from botocore.client import Config
import datetime
import numpy as np
import os
from keras.preprocessing import sequence
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout
from keras.layers import Input, LSTM
from keras.models import Model
import h5py
import pickle
from sklearn.externals import joblib # for saving minmaxscalar


####################################################################################################################################################
####################################################################################################################################################
# FUNCTIONS #
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
# BootToIndex:  Given a boolean index, converts to a numeric index
#
####################################################################################################################################################
def BoolToIndex(ll):
    these = []
    for i, l in enumerate(ll):
        if l:
            these.append(i)
    return these
####################################################################################################################################################
# batch_params: creators parameters for grouping data into scraps, training,
#               testing, and future batches
####################################################################################################################################################
def batch_params(length, batch_size, timesteps, test_percent):
    # for dividing by: scraps, train, test, future
    d =  ( { 'batch_size': batch_size, 
    'timesteps' : timesteps, 
    'test_percent' : test_percent } )

    d['future_end'] = length
    d['future_start'] = d['future_end'] - batch_size
    d['test_end'] = d['future_start']

    d['batches'] = d['test_end'] // batch_size

    d['train_start'] = d['test_end'] - batch_size * d['batches']
    d['train_end'] = int(((d['batches'] * (1 - test_percent) )//1) * batch_size + d['train_start'])
    d['test_start'] = d['train_end']

    d['scrap_end'] = d['train_start']

    return(d)
####################################################################################################################################################
# ensure_dir_exists: checks a target director to make a file, creates it
#                    if it does not exist
####################################################################################################################################################

def ensure_dir_exists(dir):
    if not os.path.exists(dir):
            os.makedirs(dir)

####################################################################################################################################################
# streamlined model: reshapes data for use with a stateful LSTM model,
#                    builds said model, and saves said model
#                    originaly just a long chunk of code that was generalized into a function
####################################################################################################################################################

# NOTE this 'function' was not originally intended to be such.  Turned into a function to make testing other features easier
def streamlined_model(data, bPar, epochs, df_name):
    # Feature Scaling
    #scale between 0 and 1. the weights are esier to find.
    from sklearn.preprocessing import MinMaxScaler

    sc = MinMaxScaler(feature_range = (0, 1))
    df_scaled = sc.fit_transform(np.float64(data) )


    X_all = []
    y_all = []

    # Creating a data structure with n timesteps
    for i in range(bPar['train_start'], bPar['future_end'] ):
        X_all.append(df_scaled[i - bPar['timesteps'] : i, 0 ] )
        if (i < bPar['future_start']):
            y_all.append(df_scaled[i : i + bPar['timesteps'], 0 ] )
    # y_all looks in the future, while x looks back.  We are butting against the end of our data, thus y_all will have to stop first

    # Reshaping: need numpy, not lists
    X_all = np.array(X_all)
    y_all = np.array(y_all)

    #X_future = np.reshape(X_future, (X_future.shape[0], X_future.shape[1], 1) )
    X_all = np.reshape(X_all, (X_all.shape[0], X_all.shape[1], 1) )
    y_all = np.reshape(y_all, (y_all.shape[0], y_all.shape[1], 1) )

    # we removed the scraps, so bPar is off by train_start / scrap_end
    X_train = X_all[ 0 : bPar['train_end'] - bPar['scrap_end'] ]
    y_train = y_all[ 0 : bPar['train_end'] - bPar['scrap_end'] ]

    # Building the LSTM
    # Importing the Keras libraries and packages
    inputs_1_mae = Input(batch_shape=(bPar['batch_size'], bPar['timesteps'],1) )
    lstm_1_mae = LSTM(10, stateful=True, return_sequences=True)(inputs_1_mae)
    dropout_1 = Dropout(0.1)(lstm_1_mae)
    lstm_2_mae = LSTM(10, stateful=True, return_sequences=True)(dropout_1)
    dropout_2 = Dropout(0.1)(lstm_2_mae)
    output_1_mae = Dense(units = 1)(dropout_2)

    regressor_mae = Model(inputs=inputs_1_mae, outputs = output_1_mae)
    regressor_mae.compile(optimizer='adam', loss = 'mae')
    regressor_mae.summary()

    #Statefull
    for i in range(epochs):
        print("Epoch: " + str(i))
        #run through all data but the cell, hidden state are used for the next batch.
        regressor_mae.fit(X_train, y_train, shuffle=False, epochs = 1, batch_size = bPar['batch_size'])
        #resets only the states but the weights, cell and hidden are kept.
        regressor_mae.reset_states()

    #save model and data

    return (regressor_mae, X_all, y_all, sc)

####################################################################################################################################################
# save_data: saves a set of parameters for reloading later
####################################################################################################################################################
def save_data(stock_name, model, X_all, y_all, bPar, sc, dates):
    this_dir = os.path.dirname(os.path.realpath('__file__') )
    this_dir = os.path.join(this_dir, 'predictions')
    this_dir = os.path.join(this_dir, stock_name)
    ensure_dir_exists(this_dir)

    f_name = stock_name + '_'  + '_model.json'
    target_dir = os.path.join(this_dir, f_name)
    model_json = model.to_json()
    with open (target_dir, 'w') as json_file:
        json_file.write(model_json)
    f_name = stock_name + '_'  + '_weights.h5'
    target_dir = os.path.join(this_dir, f_name)
    model.save_weights(target_dir)

    prefix = stock_name + '_'  + '_'
    f_name = prefix + 'X_all.npy'
    target_dir = os.path.join(this_dir, f_name)
    np.save(target_dir, X_all)

    f_name = prefix + 'y_all.npy'
    target_dir = os.path.join(this_dir, f_name)
    np.save(target_dir, y_all)

    f_name = prefix + 'bPar.pickle' # was having annoying errors--pickle works easily though
    target_dir = os.path.join(this_dir, f_name)
    with open (target_dir, 'wb') as handle:
        pickle.dump(bPar, handle, protocol=pickle.HIGHEST_PROTOCOL)

    f_name = prefix + 'sc.save'
    target_dir = os.path.join(this_dir, f_name)
    joblib.dump(sc, target_dir)

    f_name = prefix + 'dates.npy'
    target_dir = os.path.join(this_dir, f_name)
    np.save(target_dir, dates)


####################################################################################################################################################
####################################################################################################################################################
# MAIN #
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
# ETL #
####################################################################################################################################################
this_dir = os.path.dirname(os.path.realpath('__file__') )
target_dir = os.path.join(this_dir, 'target_stocks/stocks.csv')
stocks = pd.read_csv(target_dir, delimiter=',') 

print(stocks.head() )
####################################################################################################################################################
# CREATE AND SAVE MODELS #
####################################################################################################################################################
epochs = 120

# may regret the for-loop later
for stock in stocks.symbol.unique():
    these = stocks.symbol == stock
    bPar = batch_params (sum(these), batch_size = 64, timesteps = 32, test_percent = 0.1)
    these = BoolToIndex(these)
    model, X_all, y_all, sc = streamlined_model (stocks.iloc[these, 2:3].values, bPar, epochs, stock)
    stocks['date'] = pd.to_datetime(stocks['date'] )
    dates = stocks.iloc[these, 0].values
    print(len(dates))
    for i in range(bPar['timesteps']):
        date = dates[-1]
        dates = np.append(dates, date)
    save_data(stock, model, X_all, y_all, bPar, sc, dates)
