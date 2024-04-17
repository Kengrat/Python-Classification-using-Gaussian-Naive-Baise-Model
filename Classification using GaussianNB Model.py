import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # MATLAB-like way of plotting
from matplotlib.colors import Normalize, ListedColormap

# sklearn package for machine learning in python:
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB

# read data from .csv in folder
df = pd.read_csv("nba_rookie_data.csv")

#to display all columns
pd.set_option('display.max_columns', None)

# dropping the name variable
df = df.drop(["Name"], axis=1)

#print(df.head(),'\n')

X = df.iloc[:, [0, 2]].values   #input Game played and Point per Game
y = df.iloc[:, -1].values       #the last column TARGET_5Yrs

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/5, random_state=42)

## GAUSSIAN NAIVE BAYES MODEL

gnb = GaussianNB()
gnb.fit(X_train, y_train)

print("\n, For GAUSSIAN NAIVE")
# output accuracy score
print('Our Accuracy is %.2f:' % gnb.score(X_test, y_test))
# number of mislabeled points
print('Number of mislabeled points out of a total of %d points: %d' % (X_test.shape[0], (y_test != gnb.predict(X_test)).sum()))

# visualise the model
fig, ax = plt.subplots()

# need to set up a mesh to plot the contour of the model
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

# step size in the mesh
h = 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# model predicts every point in the mesh and reshapes the array for plotting
Z = gnb.predict(np.column_stack([xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

# set up the color and symbol encoding
nm = Normalize(vmin = 0, vmax = 1)
cm = ListedColormap(['blue', 'red'])
m = ['o', '^']

# contour plot of the model
ax.contourf(xx, yy, Z, cmap = cm, norm = nm, alpha=0.5)

# plot the data
for i in range(len(X_test)): ax.scatter(X_test[i,0], X_test[i,1], marker = m[gnb.predict(X_test)[i]], c = y_test[i], cmap = cm, norm = nm, s = 10)

# find the misclassified points
mis_ind = np.where(y_test != gnb.predict(X_test))[0]
# print('Misclassified Points:\n', X_test[mis_ind], y_test[mis_ind])

# plot the misclassified points
ax.scatter(X_test[mis_ind,0], X_test[mis_ind,1], marker = '*', color = 'white', s = 2)

ax.set_title('GNB Plot (Game Played VS Points per Game)')
ax.set_xlabel('Game played')
ax.set_ylabel('Points Per Game')
fig.savefig('GNB_plot.png')