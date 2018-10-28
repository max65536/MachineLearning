import numpy as np


def knn(samples, k):
    # compute density estimation from samples with KNN
    # Input
    #  samples    : DxN matrix of data points
    #  k          : number of neighbors
    # Output
    #  estDensity : estimated density in the range of [-5, 5]

    #####Insert your code here for subtask 5b#####
    # Compute the number of the samples created
    N=len(samples)
    p=np.zeros(N)
    estDensity=np.zeros((N,2))
    for i in range(N):
        pos=int(round((samples[i]+5)*10))
        tmp=abs(samples-samples[i])
        j=tmp.argsort()[k]
        p[pos]=k/(2*tmp[j]*N)

    # print(samples-samples[0])
    # print([samples,p])
    estDensity[:,0]=np.arange(-5,5.0,0.1)
    estDensity[:,1]=p
    return estDensity

def d(A,B):
    return np.square(A-B)
