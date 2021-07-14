import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re

path = r'train.csv'

data = pd.read_csv(path)
reserve_data = data.copy()
tot_rec = len(data)

# extract title from name and place it in a new column
data['Title'] = [i[1].split()[0] for i in (j.split(',') for j in (x for x in data['Name']))]
le = LabelEncoder()
data['Title'] = le.fit_transform(data['Title'])
data = data.drop(['Name'], axis=1)

# convert 'Sex' to labels
le2 = LabelEncoder()
data['Sex'] = le2.fit_transform(data['Sex'])

# fill the missing values in 'Age'
data['Age'] = data['Age'].fillna(data.groupby('Title')['Age'].transform('mean'))

# bin the ages
min_age = data['Age'].min()
max_age = data['Age'].max()
bins = [0., 18., 35., 60., 80.]
labels = [0, 1, 2, 3]
data['Age'] = pd.cut(data['Age'], bins=bins, labels=labels)

# get the family count and place it in a new column
data['Family'] = data['SibSp'] + data['Parch'] + 1
data = data.drop(['SibSp', 'Parch'], axis=1)

# bin the fare values

# split the first letter off the 'Cabin'
# data['cabin_letter'] = [re.split('[0-9]+', i, flags=re.IGNORECASE)[0] for i in data['Cabin']]

# fill the two missing values in 'Embarked'
