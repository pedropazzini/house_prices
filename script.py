import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def rmsle (predicted, actual):
    return np.sqrt(np.array([(np.log1p(predicted)-np.log1p(actual))**2]).mean())

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

y = df_train['SalePrice']
df_train = df_train.drop('SalePrice',axis=1)

df = pd.concat((df_train,df_test))
df_i = df.select_dtypes([int])
df_f = df.select_dtypes([float])
df_o = df.select_dtypes([object])

# Correct null values

# Integer columns
print(df_i.isnull().sum()) # No None values

# Object Columns
print(df_o.isnull().sum())
df_o = df_o.fillna('NAN').apply(LabelEncoder().fit_transform,axis=1)

# Float columns
print(df_f.isnull().sum()) 

df_f.GarageYrBlt[df_f.GarageArea==0] = 0
df_f.MasVnrArea[df_f.MasVnrArea.isnull()] = 198

