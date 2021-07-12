import pandas as pd
from sklearn.preprocessing import LabelEncoder

path = r'train.csv'

data = pd.read_csv(path)
reserve_data = data.copy()
tot_rec = len(data)

# extract title from name and place it in a new column
data['Title'] = [i[1].split()[0] for i in (j.split(',') for j in (x for x in data['Name']))]
le = LabelEncoder()
data['Title'] = le.fit_transform(data['Title'])

# convert 'Sex' to labels
le2 = LabelEncoder()
data['Sex'] = le2.fit_transform(data['Sex'])

# fill the missing values in 'Age'
cr = data.corr()
missing_age = data[data['Age'].isnull()]
for i in range(data['Title'].max()):
	x = round(data['Age'][data['Title'] == i].mean(), 2)
	print(i, ":", x)


# bin the ages

# get the family count and place it in a new column

# bin the fare values

# split the first letter off the 'Cabin'

# fill the two missing values in 'Embarked'
