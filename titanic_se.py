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
    lvl.append(('logreg', LogisticRegression(solver='liblinear', penalty='l2', C=10, random_state=42)))
    lvl.append(('xgb', XGBClassifier(n_estimators=19, max_depth=7, learning_rate=0.5,
                                     colsample_bytree=0.7, verbosity=0, random_state=42, use_label_encoder=False)))
    lvl.append(('rf1', RandomForestClassifier(n_estimators=1600, min_samples_split=5, min_samples_leaf=2,
                                              max_features='sqrt', max_depth=10, bootstrap=False, random_state=42)))
    lvl.append(('rf2', RandomForestClassifier(n_estimators=400, min_samples_split=10, min_samples_leaf=4,
                                              max_features='sqrt', max_depth=90, bootstrap=True, random_state=42)))
    lvl_2 = LogisticRegression(solver='liblinear', random_state=42)
    clf = StackingClassifier(estimators=lvl, final_estimator=lvl_2, cv=5)
    return clf


def eval_model(clf, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5)
    score = cross_val_score(clf, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    print(np.mean(score))
    return score

clf = stack()

clf.fit(X, y)

y_pred = clf.predict(test_set)