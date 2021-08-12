import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report

# import data from exploratory_data_analysis
from exploratory_data_analysis import data, y, test_set

# define X
X = data

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# tuning hyper-parameters
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
rand_param = {'n_estimators': n_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf,
              'bootstrap': bootstrap}

# define classifier
clf = RandomForestClassifier()

# randomized search with cross-validation
clf_random = RandomizedSearchCV(estimator=clf,
                                param_distributions=rand_param,
                                n_iter=100, cv=3, verbose=2,
                                random_state=42, n_jobs=-1)

# fit the model using the best hyper-parameters
clf_random.fit(X, y)

print(clf_random.best_params_)

y_pred = clf_random.predict(test_set)

'''
print('\nModel accuracy score: ', accuracy_score(y_test, y_pred))
print('\nConfusion Matrix\n', confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
'''