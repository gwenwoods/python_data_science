import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
from preprocess import cleanser
from sklearn.ensemble import RandomForestClassifier


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

# split and balance training data
X_fit, X_eval, y_fit, y_eval = train_test_split(X_train, y_train, test_size=0.2, random_state=1234)
X_fit_balanced, y_fit_balanced = cleanser.DataCleanser.balanceTraining(X_fit, y_fit)

# classifier & fit
    
clf = RandomForestClassifier(n_estimators=1000, max_depth=6, n_jobs=8, verbose=2, max_features=80)
clf.fit(X_fit_balanced, y_fit_balanced)

# clf = xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=200, learning_rate=0.03, nthread=8, subsample=0.8, colsample_bytree=0.85, seed=4243)
# clf.fit(X_fit_balanced, y_fit_balanced, early_stopping_rounds=20, eval_metric="auc", eval_set=[(X_eval, y_eval)])
print('Overall AUC:', roc_auc_score(y_fit_balanced, clf.predict_proba(X_fit_balanced)[:, 1]))

# predicting, use y_pred = clf.predict(X_test) for categorical prediction
y_pred = clf.predict_proba(X_test)[:, 1]

# format output data
submission = pd.DataFrame({"ID":id_test, "TARGET":y_pred})
submission.to_csv("submission_rf_0331.csv", index=False)

print('Completed!')
