# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 22:49:24 2019

@author: mbigdelou
"""
#==== Import libraries
import os
os.chdir(r'C:\Users\mbigdelou\Desktop\Machine Learning Project')

os.getcwd()

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

plt.interactive(False)

#========================
#==== Import dataset
df = pd.read_csv('bikeshare dataset.csv')

#General info about dataset
df.shape
df.describe()
df.head()

list(df)
df.info()

#checking if any value is missing
df.isnull().any()

#========================
#==== Correlation Matrix
correlation = df.corr()
plt.figure(figsize=(14, 12))
heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")
plt.show()

#========================
#==== Dropping Unwanted Variables
#since we have the categorical varibale of "season", date and time is not a relevant variable in this analysis.
df = df.drop('datetime', axis = 1)

#========================
#==== Scale Dataset,  Normalization of Variables
#Since the ranges of features are very similar, then there is no need to normalize data
# There is no need to create dummy variables (it's already created)

#========================
#==== extracting independent and target variables
X = df.drop(['count'],axis=1)
y = df['count']

#========================
#==== Scatter-plot with count

plt.scatter(df.season,y, c='r',marker='*')
plt.show()

plt.scatter(df.holiday,y, c='g',marker='*')
plt.show()

plt.scatter(df.workingday,y, c='g',marker='*')
plt.show()

plt.scatter(df.weather,y, c='g',marker='*')
plt.show()

plt.scatter(df.temp,y, c='g',marker='*')
plt.show()

plt.scatter(df.atemp,y, c='g',marker='*')
plt.show()

plt.scatter(df.humidity,y, c='g',marker='*')
plt.show()

plt.scatter(df.windspeed,y, c='g',marker='*')
plt.show()

plt.scatter(df.casual,y, c='g',marker='*')
plt.show()

plt.scatter(df.registered,y, c='g',marker='*')
plt.show()

plt.scatter(df['count'],y, c='g',marker='*')
plt.show()

plt.scatter(df.registered,df.casual, c='g',marker='*')
plt.show()

#========================
#==== Spliting dataset into traing  and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25,random_state=0) 

#========================
#==== Import Models
from sklearn.linear_model import LinearRegression
from sklearn.neighbors  import KNeighborsRegressor 
from sklearn.ensemble  import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm  import SVC

linreg = LinearRegression() 
knnreg = KNeighborsRegressor(n_neighbors=7, weights='distance') 
rfr = RandomForestRegressor(n_estimators=50, max_depth=20, random_state=0) 
adbr = AdaBoostRegressor(n_estimators=50)
svmr = SVC(kernel='poly', degree=2, gamma='scale')

#==== Train Regressor
linreg.fit(X_train,y_train)
knnreg.fit(X_train,y_train)
rfr.fit(X_train,y_train)
adbr.fit(X_train,y_train)
svmr.fit(X_train,y_train)

#==== Predict on the test set
y_pred_linreg = linreg.predict(X_test)
y_pred_knnreg = knnreg.predict(X_test)
y_pred_rfr = rfr.predict(X_test)
y_pred_adbr = adbr.predict(X_test)
y_pred_svmr = svmr.predict(X_test)


#==== Linear Regression
linreg.coef_ 
linreg.intercept_

'''
# Plot line / model
plt.scatter(y_test, y_pred_linreg)
plt.xlabel("Actual Values")
plt.ylabel("Predictions")
plt.show()
'''

import statsmodels.api as sm
X2=sm.add_constant(X_train)
ols = sm.OLS(y_train,X2) 
lr = ols.fit() 
print(lr.summary())
print ('R-Squared:', lr.rsquared, ';', 'Adj R-Squared', lr.rsquared_adj)


#==== Performance Measures
linreg.score(X_test,y_test) #R-squared for test dataset
knnreg.score(X_test,y_test)
rfr.score(X_test,y_test)
adbr.score(X_test,y_test)
svmr.score(X_test,y_test)

from sklearn.metrics import mean_squared_error
import math

mse_linreg = mean_squared_error(y_test,y_pred_linreg)
rmse_linreg = math.sqrt(mse_linreg)
print('rmse_linreg:', rmse_linreg)

mse_knnreg = mean_squared_error(y_test,y_pred_knnreg)
rmse_knnreg = math.sqrt(mse_knnreg)
print('rmse_knnreg:', rmse_knnreg)

mse_rfr = mean_squared_error(y_test,y_pred_rfr)
rmse_rfr = math.sqrt(mse_rfr)
print('rmse_rfr:', rmse_rfr)

mse_adbr = mean_squared_error(y_test,y_pred_adbr)
rmse_adbr = math.sqrt(mse_adbr)
print('rmse_adbr:', rmse_adbr)

mse_svmr = mean_squared_error(y_test,y_pred_svmr)
rmse_svmr = math.sqrt(mse_svmr)
print('rmse_svmr:', rmse_svmr)


#==== K-Folds Cross Validation (6-fold cross validation)
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics

scores_linreg = cross_val_score(linreg.fit(X_train,y_train), X_train, y_train, cv=6)
print ('Cross-validated scores linreg:', scores_linreg)

scores_knnreg = cross_val_score(knnreg.fit(X_train,y_train), X_train, y_train, cv=6)
print ('Cross-validated scores knnreg:', scores_knnreg)

scores_rfr = cross_val_score(rfr.fit(X_train,y_train), X_train, y_train, cv=6)
print ('Cross-validated scores rfr:', scores_rfr)

scores_adbr = cross_val_score(adbr.fit(X_train,y_train), X_train, y_train, cv=6)
print ('Cross-validated scores adbr:', scores_adbr)

scores_svmr = cross_val_score(svmr.fit(X_train,y_train), X_train, y_train, cv=6)
print ('Cross-validated scores svmr:', scores_svmr)


#==== Plot cross validated predictions 
predictions_linreg = cross_val_predict(linreg.fit(X_train,y_train), X_train, y_train, cv=6)
plt.scatter(y_train, predictions_linreg)
plt.show()

predictions_knnreg = cross_val_predict(knnreg.fit(X_train,y_train), X_train, y_train, cv=6)
plt.scatter(y_train, predictions_knnreg)
plt.show()

predictions_rfr = cross_val_predict(rfr.fit(X_train,y_train), X_train, y_train, cv=6)
plt.scatter(y_train, predictions_rfr)
plt.show()

predictions_adbr = cross_val_predict(adbr.fit(X_train,y_train), X_train, y_train, cv=6)
plt.scatter(y_train, predictions_adbr)
plt.show()

predictions_svmr = cross_val_predict(svmr.fit(X_train,y_train), X_train, y_train, cv=6)
plt.scatter(y_train, predictions_svmr)
plt.show()


#==== Cross-Predicted Accuracy (R-square score of the models)
accuracy_linreg = metrics.r2_score(y_train, predictions_linreg)
print ('Cross-Predicted Accuracy_linreg:', accuracy_linreg)

accuracy_knnreg = metrics.r2_score(y_train, predictions_knnreg)
print ('Cross-Predicted Accuracy_knnreg:', accuracy_knnreg)

accuracy_rfr = metrics.r2_score(y_train, predictions_rfr)
print ('Cross-Predicted Accuracy_rfr:', accuracy_rfr)

accuracy_adbr = metrics.r2_score(y_train, predictions_adbr)
print ('Cross-Predicted Accuracy_adbr:', accuracy_adbr)

accuracy_svmr = metrics.r2_score(y_train, predictions_svmr)
print ('Cross-Predicted Accuracy_svmr:', accuracy_svmr)


#==== Grid Search hyper-parameter tuning 
from sklearn.model_selection import GridSearchCV
# Linear Regression does not need hyper-parameter tuning
# KNN
model_knnreg1 = KNeighborsRegressor() 

param_dict_knnreg = {
        'n_neighbors': [4,5,6,7,9], 
        'weights': ['uniform', 'distance'], 
        'leaf_size' : [20,25,30,35,40],
        }

model_knnreg2 = GridSearchCV(model_knnreg1,param_dict_knnreg)
model_knnreg2.fit(X_train,y_train)
model_knnreg2.best_params_
model_knnreg2.best_score_

# Random Forest Regressor
model_rfr1 = RandomForestRegressor() 

param_dict_rfr = {
        'n_estimators': [20,30,40,50,60], 
        'max_depth': [10,20,30,40,50],         
        }

model_rfr2 = GridSearchCV(model_rfr1, param_dict_rfr)
model_rfr2.fit(X_train,y_train)
model_rfr2.best_params_
model_rfr2.best_score_

# AdaBoost Regressor
model_adbr1 = AdaBoostRegressor()

param_dict_adbr = {
        'n_estimators': [30,40,50,60,70],        
        'learning_rate' : [1,2,3,4],
        'loss' : ['linear','square','exponential'],         
        }

model_adbr2 = GridSearchCV(model_adbr1, param_dict_adbr)
model_adbr2.fit(X_train,y_train)
model_adbr2.best_params_
model_adbr2.best_score_

# SVC
model_svmr1 = SVC()

param_dict_svmr = {
        'gamma': ['scale'],
        'C' : [0.001,0.01,0.1,1,10],
        'kernel' : ['rbf', 'linear','poly', 'sigmoid'],        
        'degree' : [2,3,4,5]
        }

model_svmr2 = GridSearchCV(model_svmr1, param_dict_svmr, cv=None)
model_svmr2.fit(X_train,y_train)
model_svmr2.best_params_
model_svmr2.best_score_







