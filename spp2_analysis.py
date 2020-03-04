####################################################################################################################################################
####################################################################################################################################################
# tesla2.py
####################################################################################################################################################
# This is the second part of a Tesla stock price analysis program.  The first wrangled the data and fit two machine learning models to it.  As fitting
# those models leads to long runtimes, the program was split in two.
# This second part creates visualizations for the machine learning models and data in general.  It also fits and plots a regression model.
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
# IMPORTS
####################################################################################################################################################
from io import StringIO
import requests
import json
import pandas as pd
import types
from botocore.client import Config
import datetime
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import math
from keras.layers import Dense
from keras.layers import Input, LSTM
from keras.models import Model, model_from_json
import pickle
import os
from os import listdir  # used to gather names of files in folder
import h5py
from sklearn.externals import joblib # for loading the minmaxscalar
####################################################################################################################################################
####################################################################################################################################################
# FUNCTIONS #
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
# LOAD DATA: gathers up key variables from tesla.py and returns them
#            not a user-friendly function
####################################################################################################################################################
def load_data(stock_name):
    # could I just pickle everything?
    this_dir = os.path.dirname(os.path.realpath('__file__') )
    this_dir = os.path.join(this_dir, 'predictions')
    this_dir = os.path.join(this_dir, stock_name)

    prefix = stock_name + '__'

    # LOAD MODEL    
    f_name = stock_name + '__model.json'
    target_dir = os.path.join(this_dir, f_name)
    with open (target_dir, 'r') as json_file:
        model = json_file.read()
    model = model_from_json(model)
    f_name = stock_name + '__weights.h5'
    target_dir = os.path.join(this_dir, f_name)
    model.load_weights(target_dir)
    # LOAD X_all and y_all
    f_name = prefix + 'X_all.npy'
    target_dir = os.path.join(this_dir, f_name)
    X_all = np.load(target_dir)

    f_name = prefix + 'y_all.npy'
    target_dir = os.path.join(this_dir, f_name)
    y_all = np.load(target_dir)
    # LOAD bPar and sc
    f_name = prefix + 'bPar.pickle'
    target_dir = os.path.join(this_dir, f_name)
    with open (target_dir, 'rb') as handle:
        bPar = pickle.load(handle)

    f_name = prefix + 'sc.save'
    target_dir = os.path.join(this_dir, f_name)
    sc = joblib.load(target_dir)

    f_name = prefix + 'dates.npy'
    target_dir = os.path.join(this_dir, f_name)
    dates = np.load(target_dir, allow_pickle=True)

    return(model, X_all, y_all, bPar, sc, dates)

####################################################################################################################################################
# ensure_dir_exists: checks a target director to make a file, creates it
#                    if it does not exist
####################################################################################################################################################
def ensure_dir_exists(dir):
    if not os.path.exists(dir):
            os.makedirs(dir)
####################################################################################################################################################
# data_division_plot: bar plot of how data is being divided between
#                     scraps, train, test and future
####################################################################################################################################################
def data_division_plot(bPar, target_dir, title = 'Sample Sizes of Data'):
    # Visualising the Batches
    plt.bar ( ['scraps','train','test','future'], [bPar['scrap_end'], bPar['train_end'] - bPar['train_start'], bPar['test_end'] - bPar['test_start'], bPar['future_end'] - bPar['future_start'] ]  ,  label = 'Sample Sizes')
    plt.title(title)
    plt.xlabel('Samples')
    plt.ylabel('Cases')
    plt.legend()

    ensure_dir_exists(target_dir)
    target_dir = os.path.join(target_dir, 'divisions.png')
    plt.savefig(target_dir)
    plt.close()
#    plt.show()
####################################################################################################################################################
# pred_res: creates and saves visuals of predictions and residuals
#           calculates R^2
####################################################################################################################################################
def pred_res(regressor_mae, bPar, sc, X_all, y_all, target_dir, pre_name, dates):
    y_hat = regressor_mae.predict(X_all, batch_size = bPar['batch_size'])
    regressor_mae.reset_states()

    #reshaping
    y_hat = np.reshape(y_hat,(y_hat.shape[0], y_hat.shape[1] ) )
    y_all = np.reshape(y_all,(y_all.shape[0], y_all.shape[1] ) )
    X_all = np.reshape(X_all,(X_all.shape[0], X_all.shape[1] ) )

    #inverse transform  ## removing didn't do anything
    y_hat = sc.inverse_transform(y_hat)
    y_all = sc.inverse_transform(y_all)
    X_all = sc.inverse_transform(X_all)

    # make linear y_hat
    y_hat_linear30 = []
    y_hat_linear25 = []
    y_hat_linear20 = []
    y_hat_linear15 = []
    y_hat_linear10 = []
    X_plot = []

    #y_hat_linear = [0] * bPar['timesteps']
    for j in range(0, len(y_hat) ):
        X_plot = np.append(X_plot, X_all[j, 0] )
        y_hat_linear30 = np.append(y_hat_linear30, y_hat[j, bPar['timesteps'] -3 ] )
        y_hat_linear25 = np.append(y_hat_linear25, y_hat[j, bPar['timesteps'] - 8 ] )
        y_hat_linear20 = np.append(y_hat_linear20, y_hat[j, bPar['timesteps'] - 13 ] )
        y_hat_linear15 = np.append(y_hat_linear15, y_hat[j, bPar['timesteps'] - 18 ] )
        y_hat_linear10 = np.append(y_hat_linear10,  y_hat[j, bPar['timesteps'] - 23 ] ) # doing it this way for consistency...could be simpler for some of these though.

    y_hat_linear30_test = y_hat_linear30[ bPar['test_start'] - bPar['train_start'] : bPar['test_end'] - bPar['train_start'] ]
    y_test30 = X_plot[ bPar['test_start'] - bPar['train_start'] +30 : bPar['test_end'] - bPar['train_start'] + 30]
    res30 = y_test30 - y_hat_linear30_test
    R2_30 = 1 - sum( (res30) ** 2 ) / sum( (y_test30 - np.mean( y_test30 ) ) ** 2 )

    y_hat_linear25_test = y_hat_linear25[ bPar['test_start'] - bPar['train_start'] : bPar['test_end'] - bPar['train_start'] ]
    y_test25 = X_plot[ bPar['test_start'] - bPar['train_start'] +25: bPar['test_end'] - bPar['train_start'] +25 ]
    res25 = y_test25 - y_hat_linear25_test
    R2_25 = 1 - sum( (res25) ** 2 ) / sum( (y_test25 - np.mean(y_test25 ) ) ** 2 )

    y_hat_linear20_test = y_hat_linear20[ bPar['test_start'] - bPar['train_start'] : bPar['test_end'] - bPar['train_start'] ]
    y_test20 = X_plot[ bPar['test_start'] - bPar['train_start'] +20 : bPar['test_end'] - bPar['train_start'] + 20]
    res20 = y_test20 - y_hat_linear20_test
    R2_20 = 1 - sum( (res20) ** 2 ) / sum( (y_test20 - np.mean( y_test20 ) ) ** 2 )

    y_hat_linear15_test = y_hat_linear15[ bPar['test_start'] - bPar['train_start'] : bPar['test_end'] - bPar['train_start'] ]
    y_test15 = X_plot[ bPar['test_start'] - bPar['train_start'] +15: bPar['test_end'] - bPar['train_start'] +15 ]
    res15 = y_test15 - y_hat_linear15_test
    R2_15 = 1 - sum( (res15) ** 2 ) / sum( (y_test15 - np.mean(y_test15 ) ) ** 2 )

    y_hat_linear10_test = y_hat_linear10[ bPar['test_start'] - bPar['train_start'] : bPar['test_end'] - bPar['train_start'] ]
    y_test10 = X_plot[ bPar['test_start'] - bPar['train_start'] +10: bPar['test_end'] - bPar['train_start'] +10 ]
    res10 = y_test10 - y_hat_linear10_test
    R2_10 = 1 - sum( (res10) ** 2 ) / sum( (y_test10 - np.mean(y_test10 ) ) ** 2 )
    # # this would work better, be more accurate, if we were predicting changes.
    # print("R^2 for 30-day prediction")
    # print(R2_30)
    # print("R^2 for 20-day prediction")
    # print(R2_20)
    # print("R^2 for 10-day prediction")
    # print(R2_10)

    # # Visualising the results

    title = pre_name + ': Predicted vs Real Prices'

    start = bPar['train_start']
    stop = bPar['future_end'] # added a timesteps-length buffer to the end of dates--imperfect as timesteps is actually market days
    step = int( ( stop - start ) / 5 )
    plt.plot_date( dates[range(start + 10, stop + 10) ] , y_hat_linear10.astype(float), fmt = '--m', label = '10-day prediction')
    plt.plot_date( dates[range(start + 30, stop + 30) ],  y_hat_linear30.astype(float), fmt = '--g' , label = '30-day prediction')
    plt.plot_date( dates[range(start, stop)    ],  X_plot, fmt = '-r', label = 'Actual Price')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.xticks( dates[range(start, stop, step) ] )
    here = bPar['test_start']
    plt.axvline(x = dates[here])

#    plt.axvline(x = bPar['train_end'] - bPar['train_start'])
#    plt.axvline(x = bPar['test_end'] - bPar['train_start'])

    plt.text(x = dates[int((start + here)/2)], y = min(X_plot), s = 'training')
    plt.text(x = dates[here],  y = min(X_plot), s = 'testing')

    plt.legend()

    ensure_dir_exists(target_dir)
    f_name0 = pre_name + 'Predictions.png'
    target_dir0 = os.path.join(target_dir, f_name0 )
    plt.savefig(target_dir0)

    plt.close()

    start = bPar['test_start']
    stop = bPar['future_end']
    step = int( ( stop - start ) / 5 )

    title = pre_name + ': Predicted vs Real Prices-Test Data Only'
    plt.plot_date( dates[range(10 + bPar['test_start'], 10 +len(X_plot))], y_hat_linear10[bPar['test_start'] : ].astype(float), fmt = '--m', label = '10-day prediction')
    plt.plot_date( dates[range(20 + bPar['test_start'], 20 +len(X_plot))], y_hat_linear20[bPar['test_start'] : ].astype(float), fmt = '--b', label = '20-day prediction')
    plt.plot_date( dates[range(30 + bPar['test_start'], 30 +len(X_plot))], y_hat_linear30[bPar['test_start'] : ].astype(float), fmt = '--g', label = '30-day prediction')
    plt.plot_date( dates[range( 0 + bPar['test_start'],  0 +len(X_plot))], X_plot[bPar['test_start'] : ], fmt = '-r', label = 'Real Tesla Stock Price')
    plt.title(title)
    plt.xlabel('Date')
    plt.xticks( dates[range(start, stop, step) ] )
    plt.ylabel('Stock Price')

    i = bPar['test_start'] - bPar['scrap_end'] + 64
    # while i < bPar['future_end']:
    #     plt.axvline(x = i, alpha = 0.5)
    #     i+=64
    plt.legend()

    ensure_dir_exists(target_dir)
    f_name0 = pre_name + 'Predictions_Zoomed.png'
    target_dir0 = os.path.join(target_dir, f_name0 )
    plt.savefig(target_dir0)

    plt.close()

    #Residuals

    title = pre_name + ': Residuals for Stateful LSTM Model'
    plt.scatter( range(10 + bPar['test_start'], 10 + bPar['test_start'] +  len(res30) ), res30, alpha = 0.3, color = 'green' ,  label = '30-Day Residuals')
    plt.scatter( range(20 + bPar['test_start'], 20 + bPar['test_start'] +  len(res20) ), res20, alpha = 0.3, color = 'blue'  ,  label = '20-Day Residuals')
    plt.scatter( range(30 + bPar['test_start'], 30 + bPar['test_start'] +  len(res10) ), res10, alpha = 0.3, color = 'purple',  label = '10-Day Residuals')
    here = np.argmax(y_test10) + bPar['test_start']
    plt.axvline(x = here)
    plt.text(x = here, y = max(res10), s = 'local max')
    here = np.argmin(y_test10) + bPar['test_start']
    plt.axvline(x = here)
    plt.text(x = here, y = min(res10), s = 'local min')

    plt.legend()
    plt.title(title)
    plt.xlabel('Market Days After IPO')
    plt.ylabel('Residuals')

    f_name0 = pre_name + 'Residuals.png'
    target_dir0 = os.path.join(target_dir, f_name0 )
    plt.savefig(target_dir0)
    plt.close()

    title = pre_name + ': 10-Day Predicted vs Actual Values'
    plt.scatter( y_test10, y_hat_linear10_test, alpha = 0.35, label = '10-Day Predicted vs Actual')
    plt.plot( y_test10, y_test10, alpha = 0.5, color = 'grey')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    f_name0 = pre_name + 'Pred_v_actual.png'
    target_dir0 = os.path.join(target_dir, f_name0)
    plt.savefig(target_dir0)
    plt.close()

    title = pre_name + ': R-Squared Comparison'
    plt.bar ( ['10-Day','15-Day','20-Day','25-Day', '30-Day'], [R2_10, R2_15, R2_20, R2_25, R2_30]  ,  label = 'Sample Sizes')
    plt.title(title)
    plt.xlabel('Days into Future')
    plt.ylabel('R-Squared')
    plt.legend()

    target_dir0 = os.path.join(target_dir, 'R-Squared.png')
    plt.savefig(target_dir0)
    plt.close()

    return(R2_30, R2_20, R2_10)
####################################################################################################################################################
####################################################################################################################################################
# RELOADING, EXPLORATOIN, ANALYSIS, VISUALIZATION #
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
# RELOADING DATA #
####################################################################################################################################################
# this_dir = os.path.dirname(os.path.realpath('__file__') )
# target_dir = os.path.join(this_dir, 'TSLA/TSLA.csv')
# TSLA = pd.read_csv(target_dir, delimiter=',') 
# ####################################################################################################################################################
# # DESCRIBE DATA #
# ####################################################################################################################################################
# print(TSLA.head())
# print(TSLA.isnull().values.any())
# print(TSLA.describe())
# print(TSLA.dtypes)
# print(TSLA.corr())
# ####################################################################################################################################################
# # VISUALIZATION #
# ####################################################################################################################################################
# target_dir = os.path.join(this_dir, 'TSLA_Visuals')
# ensure_dir_exists(target_dir)
# target_dir = os.path.join(this_dir, 'TSLA_Visuals/General_Stock_Price.png')

# plt.plot_date(TSLA.date, TSLA.close, fmt = '-r', label = 'TESLA Stock Price')
# plt.title('TESLA stock price over time')
# plt.xticks(TSLA.date[0:len(TSLA):500])
# plt.xlabel('Date')
# plt.ylabel('Tesla price (open, high, low, close)')
# plt.legend()
# plt.savefig(target_dir)
# plt.close()

# # NEXT VISUAL #
# target_dir = os.path.join(this_dir, 'TSLA_Visuals/General_Daily_Change.png')

# plt.plot_date(TSLA.date, TSLA.daily_change, fmt = '.b', label = 'Tesla: Daily Change')
# plt.title('Tesla: Daily Price Change')
# plt.xticks(TSLA.date[0:len(TSLA):500])
# plt.xlabel('Date')
# plt.ylabel('Closing Price Change')
# plt.legend()
# plt.savefig(target_dir)
# plt.close()

# # HISTOGRAMS
# target_dir = os.path.join(this_dir, 'TSLA_Visuals/close_histograms.png')
# n_bins = 20

# target_dir = os.path.join(this_dir, 'TSLA_Visuals/close_histograms.png')

# plt.hist(TSLA.close, bins=n_bins)
# plt.title('closing')
# plt.savefig(target_dir)
# plt.close()
# target_dir = os.path.join(this_dir, 'TSLA_Visuals/daily_change_histograms.png')
# plt.hist(TSLA.daily_change, bins=n_bins)
# plt.title('daily_change')
# plt.savefig(target_dir)
# plt.close()
# ####################################################################################################################################################
# # REGRESSION ANALYSIS & VISUALIZATION #
# ####################################################################################################################################################
# x = np.array(TSLA['days_after_ipo'])
# y = np.array(TSLA['close'])

# x = np.reshape(x, (-1, 1) )
# y = np.reshape(y, (-1, 1) )
# reg = LinearRegression()

# #x = TSLA.date.reshape(-1,1)
# #y = TSLA.close.rehape(-1,1)
# #reg = reg.fit(TSLA.iloc[:,[0,4] ])
# reg.fit( x, y )

# pred = reg.predict(x)
# MSE = sum( (pred - y) ** 2) / len(x)
# R2 = 1 - sum( (pred - y) ** 2 ) / sum( (y - sum(y) / len(y) ) **2 )
# print('Regression Performance')
# print(MSE)
# print(R2)
# plt.plot(x, y, color = 'red', label = 'Real Tesla Stock Price')
# plt.plot(x, pred, color = 'green', label = 'Predicted Tesla Stock Price')
# plt.title('Real vs Predicted Tesla Stock Price')
# plt.xlabel('Market Days After IPO')
# plt.ylabel('Stock Price')
# plt.legend()
# target_dir = os.path.join(this_dir, 'TSLA_Visuals/TSLA_Closing_Reg_Predictions.png')
# plt.savefig(target_dir)
# plt.close()
# #plt.show()



this_dir = os.path.dirname(os.path.realpath('__file__') )
target_dir = os.path.join(this_dir, 'predictions')
stocks = list(listdir(target_dir) )

this_dir = os.path.join(this_dir, 'visuals')
for stock in stocks:
    model, X_all, y_all, bPar, sc, dates = load_data(stock)
    target_dir = os.path.join(this_dir, stock)
    ensure_dir_exists(target_dir)
    data_division_plot(bPar, target_dir, title = 'Sample Sizes of Data')
    pred_res(model, bPar, sc, X_all, y_all, target_dir, stock, dates)

####################################################################################################################################################
# MACHINE LEARNING ANALYISIS & VISUALIZATION #
####################################################################################################################################################
# # RELOADING
# target_dir = os.path.join(this_dir, 'TSLA_Visuals/')
# regressor_mae, X_all, y_all, bPar, sc   =   load_data('TSLA', 'closing', load_dataset = False)
# # DATA DIVISION PLOT
# data_division_plot(bPar, target_dir, title = 'Sample Sizes of Data')
# # CLOSING PRICE VISUALIZATION
# pred_res(regressor_mae, bPar, sc, X_all, y_all, target_dir, 'TSLA_Closing_Price', points = False)
# # REOADING
# regressor_mae, X_all, y_all, bPar, sc   =   load_data('TSLA', 'daily_change', load_dataset = False)
# # DAILY CHANGE VISUALIZATION
# pred_res(regressor_mae, bPar, sc, X_all, y_all, target_dir, 'TSLA_Daily_Change', points = True)

# # # export DISPLAY=localhost:0.0 (add to ~/.bashrc to make permanent)