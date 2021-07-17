from exploratory_data_analysis import data, y, test_set
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

X = np.array(data)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

clf = RandomForestClassifier(n_estimators=5)

clf.fit(X, y)

y_pred = clf.predict(test_set)
'''
print('\nModel accuracy score: ', accuracy_score(y_test, y_pred))
print('\nConfusion Matrix\n', confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
'''