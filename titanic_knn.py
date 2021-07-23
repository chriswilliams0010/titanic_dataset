from exploratory_data_analysis import data, y, test_set
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report

X = np.array(data)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

error_rate = []
for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.plot(error_rate, color='red', marker='o', markerfacecolor='red', markersize=10)

# clf = KNeighborsClassifier(n_neighbors=3)

# clf.fit(X, y)

# y_pred = clf.predict(test_set)
'''
print('\nModel accuracy score: ', accuracy_score(y_test, y_pred))
print('\nConfusion Matrix\n', confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
'''
