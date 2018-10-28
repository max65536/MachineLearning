import numpy as np
from getLogLikelihood import getLogLikelihood
from numpy.linalg import *
import matplotlib.pyplot as plt
from scipy import misc
def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out
a=np.array([[ 1.5900523  ,-1.45112395],
 [-1.45112395  ,1.32433424]])

print(inv(a))
print(abs(det(a)))
img = (misc.imread('faces.png'))
# print(img)
plt.imshow(img)
# plt.show()
print(img.shape)
print(7>8)
# f=np.zeros((10,11))
f=[[],[]]
f[0][0]=1
for i in range(10):
    for j in range(11):
        f[i,j]=f[i,j]+f[i-1,j]+f[i,j-1]+f[i-1,j-1]

print(f)
