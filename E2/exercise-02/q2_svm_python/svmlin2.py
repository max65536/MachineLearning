import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# we create 40 separable points
np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20
print('X=',X)
print('Y=',Y)

test = {}
train = {}
## Load the training data
train.update({'data': np.loadtxt('lc_train_data.dat')})
train.update({'label': np.loadtxt('lc_train_label.dat')})
test.update({'data': np.loadtxt('lc_test_data.dat')})
test.update({'label': np.loadtxt('lc_test_label.dat')})
X=train['data']
Y=train['label']
for i in range(len(Y)):
    if Y[i]==-1:
        Y[i]=0

print('X=',X)
print('Y=',Y)
# fit the model
clf = svm.SVC(kernel='linear',C=1000)
clf.fit(X, Y)

# get the separating hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(0, 1)
yy = a * xx - (clf.intercept_[0]) / w[1]

# plot the parallels to the separating hyperplane that pass through the
# support vectors
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

# plot the line, the points, and the nearest vectors to the plane
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=80, facecolors='none')
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

plt.axis('tight')
plt.show()

print('w=',clf.coef_)
print('b=',clf.intercept_)
print('sv',clf.support_vectors_)
