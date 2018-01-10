# Logistic Regression Classifier 

#**********************************#


# Imoritng the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

dataset = pd.read_csv('Social_Network_Ads.csv');
X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:,4].values



# Splitting the Data into Training Set and Testing Set
from sklearn.model_selection import train_test_split
X_Train ,X_Test , Y_Train,Y_Test= train_test_split(X,Y, test_size=0.25,random_state=0)



# Scalling the data *************
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_Train)
X_Test  =scaler.transform(X_Test)





# Fitting LOgistic Regression to The Training Set 

from sklearn import linear_model
#Prepare our Classifier*********
classifier = linear_model.LogisticRegression(C=1e5)
classifier.fit(X_Train,Y_Train);



# Predicting the Test Result*******

y_predicted = classifier.predict(X_Test)




# Making the Confusiin Matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print(accuracy_score(Y_Test,y_predicted))
cm = confusion_matrix(Y_Test,y_predicted)
print(cm)














# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, Y_Train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()




# Visualising the Test set results
'''from matplotlib.colors import ListedColormap
X_set, y_set = X_Test, Y_Test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()'''