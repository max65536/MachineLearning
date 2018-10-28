import numpy as np
from math import pi,exp,sqrt
import math as mt
def kde(samples, h):
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points ---1xN
    #  h          : (half) window size/radius of kernel
    # Output
    #  estDensity : estimated density in the range of [-5,5]

    #####Insert your code here for subtask 5a#####
    # Compute the number of samples created
    N=len(samples)
    p=np.zeros(N)
    estDensity=np.zeros((N,2))
    for i in range(N):
        pos=int(round((samples[i]+5)*10))
        # print(pos)
        for x in samples:
            p[pos]=p[pos]+kernel(x-samples[i],h)
        p[pos]=p[pos]/N

    estDensity[:,0]=np.arange(-5,5.0,0.1)
    estDensity[:,1]=p

    # print([samples,p])
    return estDensity

def kernel(u,h):
    return exp(-u*u/(2*h*h))*1/(sqrt(2*pi)*h)

# print(np.random.normal(0, 1, 100))

