from exploratory_data_analysis import data, y, test_set
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

X = np.array(data)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

clf = AdaBoostClassifier()

kf = RepeatedStratifiedKFold(n_splits=10, random_state=42)
cv_acc = cross_val_score(estimator=clf, X=X, y=y, cv=kf, scoring='accuracy')
cv_f1 = cross_val_score(estimator=clf, X=X, y=y, cv=kf, scoring='f1')
print(f" AdaBoost accuracy: \naccuracy: {cv_acc.mean()}\nstd: {cv_acc.std()}")
print(f" Adaboost f1: \nf1: {cv_f1.mean()}\nstd: {cv_f1.std()}")
