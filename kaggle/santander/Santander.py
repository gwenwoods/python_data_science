# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import math


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
    # train model
    
    rf = RandomForestClassifier(n_estimators=100)
    
    rf.fit(trainArr, trainTar)
    results = rf.predict(testArr)
   
    
    misMatch = 0
    for i in range(0, len(results)):
        if math.fabs(results[i] - trainTar[i]) > 0:
            misMatch += 1
    
    print("mismatch ", misMatch)
    
    print(type(testID), testID.shape)
    print(type(results), results.shape)
    r1 = np.reshape(results, (75818, 1))
    x = np.concatenate([testID, r1], axis=1)
    print(type(r1), r1.shape)
    print(type(x), x.shape)
    # x = pd.concat([testID, results])
    # print(type(results[0]), results[0])
    np.savetxt("foo3.csv", x, delimiter=",", fmt="%i")

    
