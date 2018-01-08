# Data Preprocessing


# Imoritng the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

dataset = pd.read_csv('Data.csv');
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values
'''print(X)
print(Y)'''


print("************************")



#Taking Care of Missing Data
from sklearn.preprocessing import Imputer 
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3]= imputer.transform(X[:,1:3])
#print(X)




print("************************")

# Encoding Ctagroical data 
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

le = preprocessing.LabelEncoder()
enc = OneHotEncoder(categorical_features=[0])

X[:,0]= le.fit_transform(X[:,0])
X = enc.fit_transform(X).toarray();
Y = le.fit_transform(Y)
#print(X)
#print(Y)

print("************************")


# Splitting the Data into Training Set and Testing Set

from sklearn.model_selection import train_test_split
X_Train ,X_Test , Y_Train,Y_Test= train_test_split(X,Y, test_size=0.2,random_state=0)

print('**************Training Data**********')

print(X_Train)
print(Y_Train)
print('**************Testing data**********')
print(X_Test)


# Feauture Scalling

#print('**************Feauture Scaling**********')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_Train)
X_Test  =scaler.transform(X_Test)

'''print(X_Train)
print(X_Test)'''