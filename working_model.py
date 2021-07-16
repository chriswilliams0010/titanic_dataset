from exploratory_data_analysis import data, y, test_set
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

test_set = pd.read_csv(r'test.csv')
test_set['Title'] = [i[1].split()[0] for i in (j.split(',') for j in (x for x in test_set['Name']))]
test_set['Title'] = le.transform(test_set['Title'])

y = np.array(data['Survived'])
X = np.array(data.iloc[:, 2:])
clf = XGBClassifier()

clf.fit(X, y)

y_pred = clf.predict(X)
