import time
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from preprocess import cleanser

para_num_run = 80
para_test_size = 0.35
para_missing = np.nan
para_n_estimator = 200
para_max_depth = 4
para_learning_rate = 0.03
para_subsample = 1.0
para_colsample_bytree = 1.0
para_early_stopping_rounds=20

###################################################
# perform one run of fit XGBoost model and prediction
def fit_and_predict(X_train, y_train):
    
    # split and balance training data
    random_seed = int(time.time()) % 10000
    X_fit, X_eval, y_fit, y_eval = train_test_split(X_train, y_train, test_size=para_test_size, random_state=random_seed)
    X_fit_balanced, y_fit_balanced = cleanser.DataCleanser.balanceTraining(X_fit, y_fit)
    X_eval_balanced, y_eval_balanced = cleanser.DataCleanser.balanceTraining(X_eval, y_eval)

    # classifier & fit
    clf = xgb.XGBClassifier(missing=para_missing, max_depth=para_max_depth, n_estimators=para_n_estimator, learning_rate=para_learning_rate,
                             nthread=8, subsample=para_subsample, colsample_bytree=para_colsample_bytree, seed=random_seed)
    clf.fit(X_fit_balanced, y_fit_balanced, early_stopping_rounds=para_early_stopping_rounds, eval_metric="auc", eval_set=[(X_eval_balanced, y_eval_balanced)])
    print('Overall AUC:', roc_auc_score(y_fit_balanced, clf.predict_proba(X_fit_balanced)[:, 1]))

    # predicting, use y_pred = clf.predict(X_test) for categorical prediction
    y_pred = clf.predict_proba(X_test)[:, 1]
    
    return y_pred

###################################################
# load data

df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

# drop constant and duplicated columns
df_train, df_test = cleanser.DataCleanser.dropConstAndDupColumns(df_train, df_test)

# drop columns not required by ML models
X_train = df_train.drop(['ID', 'TARGET'], axis=1).values
y_train = df_train['TARGET'].values

X_test = df_test.drop(['ID'], axis=1).values
id_test = df_test['ID']

y_sum = np.zeros(shape=id_test.shape)

time_start = time.time()
    
for i in range(0, para_num_run):
    print(" ------------------ Process Run " + str(i) + "  ----------------------")
    y_pred_array = fit_and_predict(X_train, y_train)
    y_sum += y_pred_array / float(para_num_run)
      
submission = pd.DataFrame({"ID":id_test, "TARGET": y_sum})
time_end = time.time()
elapsed_time = time_end - time_start

print("Total process time = " + str(elapsed_time / 60.0) + " min")

############################################################
# format output data

tstamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
submission_file_name = 'submission_multi-xgboost_' + tstamp + '.csv'
submission.to_csv(submission_file_name, index=False)

parameter_file_name = 'parameters_multi-xgboost_' + tstamp + '.log'
parameter_out = open(parameter_file_name, 'w')
parameter_out.write('test_size = ' + str(para_test_size) + '\n')
parameter_out.write('missing = ' + str(para_missing) + '\n')
parameter_out.write('n_estimator = ' + str(para_n_estimator) + '\n')
parameter_out.write('max_depth = ' + str(para_max_depth) + '\n')
parameter_out.write('learning_rate = ' + str(para_learning_rate) + '\n')
parameter_out.write('subsample = ' + str(para_subsample) + '\n')
parameter_out.write('colsample_bytree = ' + str(para_colsample_bytree) + '\n')

parameter_out.close()

print('Completed!')


