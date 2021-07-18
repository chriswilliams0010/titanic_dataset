from exploratory_data_analysis import data, y, test_set
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

X = np.array(data)


def stack():
    lvl = list()
    lvl.append(('logreg', LogisticRegression()))
    lvl.append(('rf', RandomForestClassifier()))
    lvl.append(('dt', DecisionTreeClassifier()))
    lvl_2 = RandomForestClassifier()
    clf = StackingClassifier(estimators=lvl, final_estimator=lvl_2, cv=5)
    return clf


def eval_model(clf, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)
    score = cross_val_score(clf, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    print(np.mean(score))
    return score


