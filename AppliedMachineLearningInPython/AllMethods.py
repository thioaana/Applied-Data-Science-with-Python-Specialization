import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

cancerdf = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns= np.append(cancer['feature_names'], ['target']))
X = cancerdf.drop(["target"], axis=1)
y = cancerdf.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors= 20).fit(X_train, y_train)
print("20-Neighbors Score : ", knn.score(X_test, y_test))

from sklearn.linear_model import LinearRegression
linreg = LinearRegression().fit(X_train, y_train)
print("Linear Score       : ", linreg.score(X_test, y_test))

from sklearn.linear_model import Ridge
linRidge = Ridge(alpha=10).fit(X_train, y_train)
print("Ridge (a=20) Score : ", linRidge.score(X_test, y_test))

from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2)
XPoly = poly.fit_transform(X, y)
X_train, X_test, y_train, y_test = train_test_split(XPoly, y, random_state=0)
linreg = Ridge(alpha=10).fit(X_train, y_train)
print("Poly   Score       : ", linreg.score(X_test, y_test))

from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
logreg = LogisticRegression(C=100, max_iter=10000).fit(X_train, y_train)
print("Logistic Score     : ", logreg.score(X_test, y_test))
sys.exit()

x  = np.arange(-10, 10, 0.000001)
y1 = 5+5 * x
y2 = 1/(1 + np.exp(-y1))

plt.figure()
plt.subplot(1,2,1)
plt.plot(x, y1)
plt.plot(x, y2)
plt.subplot(1,2,2)
plt.plot(x, y2)
plt.gca().set_xlim(-2, 0)
plt.gca().set_ylim(-1, 2)
plt.show()
# print(5)