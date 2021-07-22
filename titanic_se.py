from exploratory_data_analysis import data, y, test_set
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

X = np.array(data)


def stack():
    lvl = list()
    lvl.append(('logreg', LogisticRegression(solver='liblinear', penalty='l2', C=10)))
    lvl.append(('xgb', XGBClassifier(n_estimators=19, max_depth=7, learning_rate=0.5,
                                     colsample_bytree=0.7, verbosity=0)))
    lvl.append(('rf', RandomForestClassifier(n_estimators=5, max_features='sqrt', max_depth=10)))
    lvl_2 = LogisticRegression(solver='liblinear')
    clf = StackingClassifier(estimators=lvl, final_estimator=lvl_2, cv=5)
    return clf


def eval_model(clf, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5)
    score = cross_val_score(clf, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    print(np.mean(score))
    return score
