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
from sklearn.ensemble import RandomForestClassifier
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

# Adding headers to the file and converting it to csv format
def save_file_with_headers(path):
    '''
    update the file without header to add headers
    '''
    headers = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    data = pd.read_csv(path,names=headers)
    data.to_csv(path+'.csv',index=False)

def ordinal_data_classification(csvpath):
    
    cardata = pd.read_csv(csvpath)
    # encoding for each attribute to convert from qualitative to ordinal scalar sets
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
    # Normally, I would choose to do stratified sampling based on correlated attributes, however,
    # this correlation result seems counter-intuitive, so I just perform random sampling instead
    train_set, test_set = train_test_split(cardata, test_size=0.25, random_state=42)        
    
    # Random Forest Classifier
    
    # Train
    trainsetnoprice= train_set.drop('buying',axis=1)
    trainsetprice = train_set['buying'].copy()    
    
    forest_class = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42)
    forest_class.fit(trainsetnoprice, trainsetprice)
    # Test 
    testsetnoprice= test_set.drop('buying',axis=1)
    testsetprice = test_set['buying'].copy()
    
    testsetpricepredRF= forest_class.predict(testsetnoprice)
    trainsetpricepredRF= forest_class.predict(trainsetnoprice)
    
    #Performance Metrics (MSE, RMSE)
    lin_mse_trainRF = mean_squared_error(trainsetprice, trainsetpricepredRF)
    lin_mse_testRF = mean_squared_error(testsetprice, testsetpricepredRF)
    lin_rmse_trainRF = np.sqrt(lin_mse_trainRF)
    lin_rmse_testRF = np.sqrt(lin_mse_testRF)
    print('mse_trainRF:',lin_mse_trainRF)
    print('rmse_trainRF:',lin_rmse_trainRF)
    print('mse_testRF:',lin_mse_testRF)
    print('rmse_testRF:',lin_rmse_testRF)
    
    # Since for Random Forest Regressor RMSE_train < RMSE_test, we can conclude that there is some overfitting of the data

    #Test on triple row dataframe (since we do not know the capacity, we test for all three available categories)
    clientrequest1 = pd.DataFrame([['2','2','0', '2', '2','2'],['2','2','1', '2', '2','2'],['2','2','2', '2', '2','2']], columns=['maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])
#    clientrequest = np.array([2,2,2,2,2,2])
#    clientrq = clientrequest.reshape(1,-1)
    priceprediction1RF = forest_class.predict(clientrequest1)
    print('Capacity of 2,4,more:',priceprediction1RF)
    pricepredictions = {'From RF:':priceprediction1RF}
    # Results for Random Forest Classifier indicate low buying price.
    return pricepredictions

rawpath = 'car.data'
csvpath = 'car.data.csv'

#save_file_with_headers(rawpath)
car_data_eval = ordinal_data_classification(csvpath)