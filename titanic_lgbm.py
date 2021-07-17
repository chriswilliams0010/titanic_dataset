from exploratory_data_analysis import data, y, test_set
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgbm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

X = np.array(data)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
train_data = lgbm.Dataset(data=X_train, label=y_train, free_raw_data=False)
test_data = lgbm.Dataset(data=X_test, label=y_test, free_raw_data=False)
param_grid = {
    'boosting': 'dart',
    'application': 'binary',
    'learning_rate': 0.05,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.7,
    'num_leaves': 41,
    'metric': 'binary_logloss',
    'drop_rate': 0.15
}
eval_result = {}
clf = lgbm.train(train_set=train_data,
                 params=param_grid,
                 valid_sets=[train_data, test_data],
                 valid_names=['Train', 'Test'],
                 evals_result=eval_result,
                 num_boost_round=500,
                 early_stopping_rounds=100,
                 verbose_eval=20)
optimum_boost_rounds = clf.best_iteration
'''xgb = RandomizedSearchCV(param_distributions=param_grid,
                                estimator=clf, scoring='accuracy',
                                verbose=1, n_iter=50, cv=4)

xgb.fit(X, y)
print("Best params: ", xgb.best_params_)
print("Best accuracy: ", xgb.best_score_)

y_pred = xgb.predict(test_set)
'''
# print('\nModel accuracy score: ', accuracy_score(y_test, y_pred))
# print('\nConfusion Matrix\n', confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
y_pred = clf.predict(test_set)
