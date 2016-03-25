# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
import math
import xgboost as xgb


if __name__ == '__main__':
    
    print("Create Training Data:")
    
    train = pd.read_csv('data/train.csv')
    header = train.columns.values
    print(header)
    print(train.shape)
    
    train_1 = train[train['TARGET'] == 1]
    train_0 = train[train['TARGET'] == 0]
    print("train_1" , train_1.shape, type(train_1))
    print("train_0" , train_0.shape)
    
    train_1_upsample = pd.concat([train_1] * 24)
    print("train_1_upsample" , train_1_upsample.shape)
    
    train_balanced = pd.concat([train_0, train_1_upsample])
    print("train_balanced" , train_balanced.shape)
    
    numeric_cols = train._get_numeric_data().columns
    print(len(train_balanced.columns))
    print(len(numeric_cols))
    
    inputs = header[1:370]
    target = header[370:371]

    trainArr = train_balanced.as_matrix(inputs)  # training array
    trainTar = train_balanced.as_matrix(target).ravel()  # training targets
    
    #----------------------------------------
    test = pd.read_csv('data/test.csv')
    testArr = test.as_matrix(inputs)
    testID = test.as_matrix(header[0:1])
    
    #----------------------------------------
    # Random Forest model
    
    # rf = RandomForestClassifier(n_estimators=100)
    # rf.fit(trainArr, trainTar)
    # results_rf = rf.predict(testArr)
    
    #---------------------------
    # GBM models
    
    # gbm = GradientBoostingClassifier(n_estimators=40).fit(trainArr, trainTar)
    # results_gbm = gbm.predict(testArr)
    
    #---------------------------
    # XGBoost
    
    xgbm = xgb.XGBClassifier(n_estimators=200).fit(trainArr, trainTar)
    results_xgbm = xgbm.predict(testArr)

    #---------------------------
    # Output the prediction
    
    model_result = results_xgbm
    output_name = "results_xgbm_n200.csv"
    
    result = np.reshape(model_result, (75818, 1))
    x = np.concatenate([testID, result], axis=1)
   
    print(type(result), result.shape)
    print(type(x), x.shape)

    np.savetxt(output_name, x, delimiter=",", header="ID,TARGET", comments="", fmt="%i")

    
