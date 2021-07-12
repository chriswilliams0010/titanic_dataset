import pandas as pd

path = r'train.csv'

data = pd.read_csv(path)
reserve_data = data.copy()
tot_rec = len(data)
missing = data.info(verbose=True)

# extract title from name and place it in a new column
data['Title'] = [i[1].split()[0] for i in (j.split(',') for j in (x for x in data['Name']))]

# fill the missing values in 'Age'
# bin the ages

# get the family count and place it in a new column

# bin the fare values

# split the first letter off the 'Cabin'

# fill the two missing values in 'Embarked'
