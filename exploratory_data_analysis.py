import pandas as pd
# import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
# import seaborn as sns

path = r'train.csv'
test_path = r'test.csv'

data = pd.read_csv(path)
reserve_data = data.copy()
y = data.pop('Survived')
ohe = OneHotEncoder()
mm = MinMaxScaler()
scl = StandardScaler()

test_data = pd.read_csv(test_path)
pass_id = test_data['PassengerId']
data = pd.concat([data, test_data])
reserve_dataset = data.copy()

data = data.drop(['PassengerId'], axis=1)

# one hot encode 'Pclass'
one_df = ohe.fit_transform(data[['Pclass']]).toarray()
data[['Pclass_1', 'Pclass_2', 'P_class_3']] = one_df

# extract title from name and place it in a new column
data['Title'] = data['Name'].str.extract('([A-Za-z]+)\.', expand=True)

# fill the missing values in 'Age'
data['Age'] = data['Age'].fillna(data.groupby('Title')['Age'].transform('mean'))

data = pd.get_dummies(data, columns=['Title'])
data = data.drop(['Name'], axis=1)

# convert 'Sex' to labels
data = pd.get_dummies(data, columns=['Sex'])

'''
# bin the ages
bins_age = [0., 18., 35., 60., 80.]
labels_age = [0, 1, 2, 3]
data['Age'] = pd.cut(data['Age'], bins=bins_age, labels=labels_age)
'''

# get the family count and place it in a new column
data['Family'] = data['SibSp'] + data['Parch'] + 1
data = data.drop(['SibSp', 'Parch'], axis=1)

# drop 'Ticket'
data = data.drop(['Ticket'], axis=1)

# bin the fare values
data['Fare'] = (data['Fare'].fillna(data.groupby('Pclass')['Fare'].transform('mean')))
# bins_fare = [-np.inf, 50., 200., np.inf]
# labels_fare = [0, 1, 2]
# data['Fare'] = pd.cut(data['Fare'], bins=bins_fare, labels=labels_fare)

data = data.drop(['Pclass'], axis=1)

# drop 'Cabin'
data = data.drop(['Cabin'], axis=1)

# fill the two missing values in 'Embarked'
data['Embarked'] = data['Embarked'].fillna(method='ffill')
data = pd.get_dummies(data, columns=['Embarked'])

# scale 'Age' 'Fare' and 'Family'
data = scl.fit_transform(data)
# data[['Age', 'Fare']] = mm.fit_transform(data[['Age', 'Fare']])
# data[['Family']] = scl.fit_transform(data[['Family']])
# test_set = data.loc[data['set'] == 'test']
# data = data.loc[data['set'] == 'train']
# data = data.drop(['PassengerId'], axis=1)
# test_set = test_set.drop(['PassengerId'], axis=1)
# sp_corr = data.corr(method='spearman')
# sns.heatmap(sp_corr)

# sns.boxplot(data=data.iloc[:, 1:])
test_set = data[891:, :]
data = data[:891, :]
