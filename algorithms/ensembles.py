from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import numpy as np
import copy

def ensemble_forest(X, y):
    forests = []
    for init_state in range(0, 40):
        reg = RandomForestRegressor(max_depth=2, n_estimators=100, random_state=init_state)
        forest = reg.fit(X, y)
        forests.append(copy.copy(forest))
    return forests


def ensemble_ridge(X, y):
    models = []
    for init_state in range(0, 40):
        reg = Ridge(alpha=0.5, random_state=init_state)
        model = reg.fit(X, y)
        models.append(copy.copy(model))
    return models


def models_predict(models, X_in):
    ys = []
    for model in models:
        y = model.predict(X_in)
        ys.append(copy.copy(y))

    return np.array([sum(x) * (1 / len(models)) for x in zip(*ys)])


#def models_predict_rank(models, X_in):
#    ys = []
#    for model in models:
#        y = model.predict(X_in)
#        ys.append(copy.copy(my_rank(y)))
#
#    return np.array([sum(x) * (1 / len(models)) for x in zip(*ys)])

