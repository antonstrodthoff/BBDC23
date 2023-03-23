import pandas as pd
import numpy as np
from scipy import stats

def removeOutliers(data, zScore=3):
    return data[(np.abs(stats.zscore(data)) < zScore).all(axis=1)]


# function that removes the outliers of only one column
def removeOutliersColumn(data, column, zScore=3):
    return data[(np.abs(stats.zscore(data.iloc[:, column])) < zScore).all(axis=1)]
    

df = pd.DataFrame(np.array([["kjh", "ikgh", "KJhkjh", "kjhbkj", "kjhg", "ölkj", "lkj", "ppouiz", "sedrff", "okjhv", "Kjvs"],[0,1,2,3,4,5,3,2,4,1,30]]).T)
df.iloc[:, 1] = df.iloc[:, 1].astype(float)
print(df.info())
print(df)

q = df.iloc[:, 1].quantile(0.99)
print(q)
print(df[df.iloc[:, 1] < q])
#print(removeOutliersColumn(df, 1))
