'''
@author: wen
'''

import numpy as np
import pandas as pd

class DataCleanser(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
    
    @staticmethod
    def dropConstAndDupColumns(df_train, df_test):
        # remove constant columns
        remove = []
        for col in df_train.columns:
            if df_train[col].std() == 0:
                remove.append(col)
        
        df_train.drop(remove, axis=1, inplace=True)
        df_test.drop(remove, axis=1, inplace=True)
        
        # remove duplicated columns
        remove = []
        c = df_train.columns
        for i in range(len(c) - 1):
            v = df_train[c[i]].values
            for j in range(i + 1, len(c)):
                if np.array_equal(v, df_train[c[j]].values):
                    remove.append(c[j])

        df_train.drop(remove, axis=1, inplace=True)
        df_test.drop(remove, axis=1, inplace=True)

        return df_train, df_test
    
    @staticmethod
    def balanceTraining(X_fit, y_fit):
        df_fit = pd.DataFrame(np.concatenate([X_fit, np.reshape(y_fit, (len(y_fit), 1))], axis=1))

        col_num = len(X_fit[0])
        df_fit_1 = df_fit[df_fit[col_num] == 1]
        df_fit_0 = df_fit[df_fit[col_num] == 0]
        
        count_1 = len(df_fit_1)
        count_0 = len(df_fit_0)

        df_fit_balanced = pd.DataFrame
        
        if count_0 > count_1:
            factor = int(round(count_0 / float(count_1)))
            df_fit_1_upsample = pd.concat([df_fit_1] * factor)
            df_fit_balanced = pd.concat([df_fit_0, df_fit_1_upsample])
        else:
            factor = int(round(count_1 / float(count_0)))
            df_fit_0_upsample = pd.concat([df_fit_0] * factor)
            df_fit_balanced = pd.concat([df_fit_1, df_fit_0_upsample])
    
        df_fit_balanced = df_fit_balanced.iloc[np.random.permutation(len(df_fit_balanced))]
        print("df_fit_balanced shape: ", df_fit_balanced.shape)
        df_header = df_fit_balanced.columns.values
    # print("fit header ", df_fit_balanced.columns.values )
    
        X_fit_balanced = df_fit_balanced.as_matrix(df_header[0:col_num])
        y_fit_balanced = df_fit_balanced.as_matrix(df_header[col_num:col_num + 1]).ravel()
        
        return X_fit_balanced, y_fit_balanced
