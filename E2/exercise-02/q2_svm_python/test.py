import numpy as np
from kern import kern
# import os # might need to add path to mingw-w64/bin for cvxopt to work
# # os.environ["PATH"] += os.pathsep + ...
# os.environ["path"]+='C:\\Users\\lz\\AppData\\Local\\Programs\\Python\\Python35\\Lib\\site-packages\\cvxopt'
import cvxopt
X=np.arange(10).reshape(2,5)
Xi=np.hsplit(X, 5)
print('Xi=',Xi)
print('X2=',Xi[2])
print('X1=',Xi[1])
res=kern(Xi[2],Xi[1],2)
print('res=',res)
# X=np.zeros((6,5), dtype="b")
# X[0,0]=True
# print(X)
# Xi=np.hsplit(X, 5)
# print(Xi)
# C=10
# A=cvxopt.matrix(np.hstack([np.zeros(5),20*np.ones(5)]))
# A = cvxopt.matrix([1., 2.], (1,4))
# print(A)
# G=np.vstack([-np.eye(5),np.eye(5)])
# print(G)

