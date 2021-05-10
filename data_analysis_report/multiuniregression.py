# -*- coding: utf-8 -*-
"""
Created on Mon May 10 10:15:00 2021

@author: dynamit
"""

# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Other imports
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# To plot pretty figures

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
rootdir = "."
taskid = "regression"
figpath = os.path.join(rootdir, "figs", taskid)
os.makedirs(figpath, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(figpath, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def save_file_with_headers(path):
    '''
    update the file without header to add headers
    '''
    headers = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    data = pd.read_csv(path,names=headers)
    data.to_csv(path+'.csv',index=False)

def ordinal_data_regression(csvpath):
    
    cardata = pd.read_csv(csvpath)
    # encoding for each attribute
    cardata['buying'] = cardata['buying'].astype('category')
    cardata['buying'] = cardata['buying'].cat.reorder_categories(['low', 'med', 'high', 'vhigh'], ordered=True)
    cardata['buying'] = cardata['buying'].cat.codes
    cardata['maint'] = cardata['maint'].astype('category')
    cardata['maint'] = cardata['maint'].cat.reorder_categories(['low', 'med', 'high', 'vhigh'], ordered=True)
    cardata['maint'] = cardata['maint'].cat.codes
    cardata['doors'] = cardata['doors'].astype('category')
    cardata['doors'] = cardata['doors'].cat.reorder_categories(['2', '3', '4', '5more'], ordered=True)
    cardata['doors'] = cardata['doors'].cat.codes
    cardata['persons'] = cardata['persons'].astype('category')
    cardata['persons'] = cardata['persons'].cat.reorder_categories(['2', '4', 'more'], ordered=True)
    cardata['persons'] = cardata['persons'].cat.codes
    cardata['lug_boot'] = cardata['lug_boot'].astype('category')
    cardata['lug_boot'] = cardata['lug_boot'].cat.reorder_categories(['small', 'med', 'big'], ordered=True)
    cardata['lug_boot'] = cardata['lug_boot'].cat.codes
    cardata['safety'] = cardata['safety'].astype('category')
    cardata['safety'] = cardata['safety'].cat.reorder_categories(['low', 'med', 'high'], ordered=True)
    cardata['safety'] = cardata['safety'].cat.codes
    cardata['class'] = cardata['class'].astype('category')
    cardata['class'] = cardata['class'].cat.reorder_categories(['unacc', 'acc', 'good', 'vgood'], ordered=True)
    cardata['class'] = cardata['class'].cat.codes
    
    # checking for attribute correlation using Pearson's coefficient and scatter matrix
    
    corr_matrix = cardata.corr()
    pricecor = corr_matrix['buying'].sort_values(ascending=False)
    print(pricecor)
    attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    pd.plotting.scatter_matrix(cardata[attributes], figsize=(12, 8))
    save_fig("scatter_matrix_plot")
    
    # splitting the dataset into a training set and a test set
    # I noticed that there is a slight negative correlation between buying price and car acceptability
    # Normally, I would choose to do stratified sampling based on correlated attributes, however
    # this result seems counter-intuitive, so I just perform random sampling instead
    
    train_set, test_set = train_test_split(cardata, test_size=0.2, random_state=42)        
#    train_set, test_set = train_test_split(cardata, test_size=0.05, random_state=42)
    
    # Linear Regression, Decision Tree & Random Forest Regressor
    
    # Train
    trainsetnoprice= train_set.drop('buying',axis=1)
    trainsetprice = train_set['buying'].copy()    
    lin_reg = LinearRegression()
    lin_reg.fit(trainsetnoprice, trainsetprice)
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(trainsetnoprice, trainsetprice)
    forest_reg = RandomForestRegressor()
    forest_reg.fit(trainsetnoprice, trainsetprice)
    # Test 
    testsetnoprice= test_set.drop('buying',axis=1)
    testsetprice = test_set['buying'].copy()
    testsetpricepredLR= lin_reg.predict(testsetnoprice)
    trainsetpricepredLR= lin_reg.predict(trainsetnoprice)
    testsetpricepredDT= tree_reg.predict(testsetnoprice)
    trainsetpricepredDT= tree_reg.predict(trainsetnoprice)
    testsetpricepredRF= forest_reg.predict(testsetnoprice)
    trainsetpricepredRF= forest_reg.predict(trainsetnoprice)
    
    
    #Performance Metrics (MSE, RMSE)
    lin_mse_trainLR = mean_squared_error(trainsetprice, trainsetpricepredLR)
    lin_mse_testLR = mean_squared_error(testsetprice, testsetpricepredLR)
    lin_rmse_trainLR = np.sqrt(lin_mse_trainLR)
    lin_rmse_testLR = np.sqrt(lin_mse_testLR)
    lin_mse_trainDT = mean_squared_error(trainsetprice, trainsetpricepredDT)
    lin_mse_testDT = mean_squared_error(testsetprice, testsetpricepredDT)
    lin_rmse_trainDT = np.sqrt(lin_mse_trainDT)
    lin_rmse_testDT = np.sqrt(lin_mse_testDT)
    lin_mse_trainRF = mean_squared_error(trainsetprice, trainsetpricepredRF)
    lin_mse_testRF = mean_squared_error(testsetprice, testsetpricepredRF)
    lin_rmse_trainRF = np.sqrt(lin_mse_trainRF)
    lin_rmse_testRF = np.sqrt(lin_mse_testRF)
    print('mse_trainLR:',lin_mse_trainLR,'mse_trainDT:',lin_mse_trainDT, 'mse_trainRF:',lin_mse_trainRF)
    print('rmse_trainLR:',lin_rmse_trainLR,'rmse_trainDT:',lin_rmse_trainDT,'rmse_trainRF:',lin_rmse_trainRF)
    print('mse_testLR:',lin_mse_testLR,'mse_testDT:',lin_mse_testDT,'mse_testRF:',lin_mse_testRF)
    print('rmse_testLR:',lin_rmse_testLR,'rmse_testDT:',lin_rmse_testDT,'rmse_testRF:',lin_rmse_testRF)
    # Since for Linear Regression RMSE_train > RMSE_test, we can conclude that there is a slight underfitting of the data
    # Since for Decision Tree Regressor RMSE_train < RMSE_test, we can conclude that there is overfitting of the data
    # Since for Random Forest Regressor RMSE_train < RMSE_test, we can conclude that there is some overfitting of the data
    #Test on single row dataframe (since we do not know the capacity, we test for all three available categories)
    clientrequest1 = pd.DataFrame([['2','2','0', '2', '2','2'],['2','2','1', '2', '2','2'],['2','2','2', '2', '2','2']], columns=['maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])
    priceprediction1LR = lin_reg.predict(clientrequest1)
    priceprediction1DT = tree_reg.predict(clientrequest1)
    priceprediction1RF = forest_reg.predict(clientrequest1)
    print('Capacity of 2:',priceprediction1LR,priceprediction1DT,priceprediction1RF)
    # Results for Linear Regression seem to indicate med buying price for the car, the higher the car capacity
    # Results for Decision Tree and Random Forest seem to indicate low buying price, but I believe some debugging is needed.
    return clientrequest1

rawpath = 'car.data'
csvpath = 'car.data.csv'

#save_file_with_headers(rawpath)
car_data_eval = ordinal_data_regression(csvpath)