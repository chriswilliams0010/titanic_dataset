import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from exploratory_data_analysis import data, y, test_set

X = np.array(data)
X_test = np.array(test_set)

# create dict of algorithms
models = {
    'Logistic Regression': LogisticRegression(solver='liblinear', penalty='l2', C=10, random_state=42),
    'RFE': RFE(estimator=LogisticRegression()),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest 1': RandomForestClassifier(n_estimators=1600, min_samples_split=5, min_samples_leaf=2,
                                            max_features='sqrt', max_depth=10, bootstrap=False, random_state=42),
    'Random Forest 2': RandomForestClassifier(n_estimators=400, min_samples_split=5, min_samples_leaf=2,
                                            max_features='sqrt', max_depth=10, bootstrap=False, random_state=42),
    'Bagging': BaggingClassifier(base_estimator=AdaBoostClassifier()),
    'AdaBoost': AdaBoostClassifier(),
    'Extra Trees': ExtraTreesClassifier()
}
# models = {'xgb': XGBClassifier()}
# use k-fold cv to determine best estimator
# make a loop for testing and printing out results of each model
for name, model in models.items():
    kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    cv = cross_val_score(estimator=model, X=X, y=y, cv=kf, scoring='accuracy')
    print(f" {name}: \nf1: {cv.mean()}\nstd: {cv.std()}")
