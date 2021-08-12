from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report

# import data from exploratory_data_analysis
from exploratory_data_analysis import data, y

# define X
X = data

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# define classifier
clf = DecisionTreeClassifier()

# tuning hyper-parameters
criterion = ['gini', 'entropy']
max_depth = [2, 4, 6, 8, 10, 12]
max_features=[1, 'auto', 'sqrt', 'log2', None]
min_samples_leaf=[1, 2, 4, 6, 8]
params = dict(criterion=criterion, max_depth=max_depth,
              max_features=max_features, min_samples_leaf=min_samples_leaf)

# grid search cross-validation
clf_gs = GridSearchCV(clf, params)

# fit the model
clf_gs.fit(X, y)

# print out results
print('Best criterion: ', clf_gs.best_estimator_.get_params()['criterion'])
print('Best max_depth: ', clf_gs.best_estimator_.get_params()['max_depth'])
print('Best max_features: ', clf_gs.best_estimator_.get_params()['max_features'])
print('Best min_samples_leaf: ', clf_gs.best_estimator_.get_params()['min_samples_leaf'])

y_pred = clf_gs.predict(X_test)

# print('\nModel accuracy score: ', accuracy_score(y_test, y_pred))
# print('\nConfusion Matrix\n', confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
