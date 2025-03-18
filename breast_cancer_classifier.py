import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


data=pd.read_csv("wdbc.data",header=None)
data.columns=['ID','Diagnosis']+[f'Features{i}' for i in range(1,31)]

data['Diagnosis']=data['Diagnosis'].map({'B':0, 'M':1})

X=data.iloc[:,2:].values
y=data['Diagnosis'].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

prior_benign=np.mean(y_train==0)
prior_malign=np.mean(y_train==1)

print(f"Prior probability of benign:{prior_benign}")
print(f"Prior probability of malign: {prior_malign}")

from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix

# Calculate mean and covariance for each class
mean_benign = X_train[y_train == 0].mean(axis=0)
mean_malign = X_train[y_train == 1].mean(axis=0)
cov_benign = np.cov(X_train[y_train == 0], rowvar=False)
cov_malign = np.cov(X_train[y_train == 1], rowvar=False)

# Classify using MAP
y_pred = []
for x in X_test:
    p_benign = multivariate_normal.pdf(x, mean=mean_benign, cov=cov_benign) * prior_benign
    p_malign = multivariate_normal.pdf(x, mean=mean_malign, cov=cov_malign) * prior_malign
    y_pred.append(1 if p_malign > p_benign else 0)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)


'''
data=np.loadtxt(file_path,delimiter=',',dtype=object)
ID=data[:,0]
diagnosis=data[:,1]
features= data[:,2:].astype(float)

print("IDs:",ID[:5])
print("Diagnosis:",diagnosis[:5])
print("Features:",features[:5])
'''