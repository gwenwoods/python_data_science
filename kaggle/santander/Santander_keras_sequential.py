import time
import datetime
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from preprocess import cleanser
from sklearn import preprocessing

para_layer_1_out_dim = 64
para_layer_1_init = "glorot_uniform"
para_layer_1_activation = "relu"

para_layer_2_out_dim = 1
para_layer_2_init = "glorot_uniform"
para_layer_2_activation = "sigmoid"

para_batch_size = 10000

#---------------------------------
# Load datasets

df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

df_train, df_test = cleanser.DataCleanser.dropConstAndDupColumns(df_train, df_test)
# print("df_train", df_train.shape)

#---------------------------------
# Prepare data for model fits

row_num = (df_train.shape)[0]
col_num = (df_train.shape)[1]

header = df_train.columns.values
print(header)

X_fit = df_train.as_matrix(header[1:col_num - 1])  # training array
y_fit = df_train.as_matrix(header[col_num - 1:col_num])

X_fit_balanced, y_fit_balanced = cleanser.DataCleanser.balanceTraining(X_fit, y_fit)
scaler = preprocessing.StandardScaler().fit(X_fit)
X_fit_balanced_nor = scaler.transform(X_fit_balanced)  

#----------------------------------
# Prepare testing data
id_test = df_test['ID']
X_test = df_test.as_matrix(header[1:col_num - 1])
X_test_nor = scaler.transform(X_test) 

#----------------------------------
# Initiate model type
model = Sequential()
model.add(Dense(output_dim=para_layer_1_out_dim, input_dim=(col_num - 2), init=para_layer_1_init))
model.add(Activation(para_layer_1_activation))
model.add(Dense(output_dim=1, init=para_layer_2_init))
model.add(Activation(para_layer_2_activation))
model.compile(loss='binary_crossentropy', optimizer='sgd')

#-----------------------------------
# Fit and prediction
model.fit(X_fit_balanced_nor, y_fit_balanced, nb_epoch=15, batch_size=para_batch_size)
y_pred = model.predict(X_test_nor, batch_size=para_batch_size).ravel()

submission = pd.DataFrame({"ID":id_test, "TARGET":y_pred})

tstamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
submission_file_name = 'submission_keras_' + tstamp + '.csv'
submission.to_csv(submission_file_name, index=False)

print('Completed!')


