import numpy as np
from getLogLikelihood import getLogLikelihood
from MStep import MStep
from regularize_cov import regularize_cov
from estGaussMixEM import estGaussMixEM
import matplotlib.pyplot as plt
from plotModes import plotModes
from mpl_toolkits.mplot3d import Axes3D
# load datasets
epsilon=0.0001
K=3
n_iter = 5
idx=0
print('\n')
# data = [[], [], []]
# data[0] = np.loadtxt('data1')
# data[1] = np.loadtxt('data2')
# data[2] = np.loadtxt('data3')

# weights, means, covariances = estGaussMixEM(data[idx], K, n_iter, epsilon)

# # plot result
# plt.subplot()
# plotModes(np.transpose(means), covariances, data[idx])
# plt.title('Data {0}'.format(idx+1))
# plt.show()

print('(g) performing skin detection with GMMs')
sdata = np.loadtxt('skin.dat')
ndata = np.loadtxt('non-skin.dat')
# nweights, nmeans, ncovariances = estGaussMixEM(ndata[:,[0,2]], K, n_iter, epsilon)
# # sweights, smeans, scovariances = estGaussMixEM(sdata[:,[1,2]], K, n_iter, epsilon)
# # plot result
# plt.subplot()
# plotModes(np.transpose(nmeans), ncovariances, ndata[:,[0,2]])
# plt.title('Data skin')
# plt.show()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sdata[:,0],sdata[:,1],sdata[:,2])
# ax.scatter(ndata[:,0],ndata[:,1],ndata[:,2])
ax.set_xlabel('R Label')
ax.set_ylabel('G Label')
ax.set_zlabel('B Label')
plt.title('Data skin')
plt.show()
