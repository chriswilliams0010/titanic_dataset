import pandas as pd
from titanic_knn import y_pred as a
from titanic_lgbm import y_pred as b
from titanic_rf import y_pred as c
from titanic_xgb import y_pred as d
from titanic_dt import y_pred as e

df = pd.concat([pd.Series(a), pd.Series(round(b)).astype('int'), pd.Series(c), pd.Series(d), pd.Series(e)], axis=1)
