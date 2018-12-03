import numpy as np

A=np.arange(10).reshape(5,2)
w=np.arange(5)

print('A=',A)
print('w=',w)
print('sum=',w.sum())
print(np.multiply(A[:,0],w))

X=np.vstack((np.multiply(A[:,0],w),np.multiply(A[:,1],w)))
print('X=\n',X.T)


