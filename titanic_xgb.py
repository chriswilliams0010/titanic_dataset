from exploratory_data_analysis import data, y, test_set
import numpy as np
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

X = np.array(data)

# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
param_grid = {
    'n_estimators': range(5, 20),
    'max_depth': range(6, 10),
    'learning_rate': [0.4, 0.45, 0.5, 0.55, 0.6],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1]
}
clf = XGBClassifier(n_estimators=10)
xgb = RandomizedSearchCV(param_distributions=param_grid,
                                estimator=clf, scoring='accuracy',
                                verbose=1, n_iter=50, cv=4)

xgb.fit(X, y)
print("Best params: ", xgb.best_params_)
print("Best accuracy: ", xgb.best_score_)

y_pred = xgb.predict(test_set)

# print('\nModel accuracy score: ', accuracy_score(y_test, y_pred))
# print('\nConfusion Matrix\n', confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
