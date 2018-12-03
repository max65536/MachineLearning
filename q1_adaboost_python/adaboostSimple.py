import numpy as np
from numpy.random import choice
from plot_ import plot_
import matplotlib.pyplot as plt

from simpleClassifier import simpleClassifier
def adaboostSimple(X, Y, K, nSamples):
    # Adaboost with decision stump classifier as weak classifier
    #
    # INPUT:
    # X         : training examples (numSamples x numDim)
    # Y         : training lables (numSamples x 1)
    # K         : number of weak classifiers to select (scalar)
    #             (the _maximal_ iteration count - possibly abort earlier
    #              when error is zero)
    # nSamples  : number of training examples which are selected in each round (scalar)
    #             The sampling needs to be weighted!
    #             Hint - look at the function 'choice' in package numpy.random
    #
    # OUTPUT:
    # alphaK 	: voting weights (K x 1) - for each round
    # para		: parameters of simple classifier (K x 2) - for each round
    #           : dimension 1 is j
    #           : dimension 2 is theta

    #####Insert your code here for subtask 1c#####
    numSamples=len(Y)
    alphaK=np.ones(K)
    para=np.zeros((K,2))
    I=np.zeros(numSamples,dtype=int)
    w=np.ones(numSamples)/numSamples
    for m in range(K):
        # newX=np.vstack((np.multiply(X[:,0],w),np.multiply(X[:,1],w)))
        j,theta=simpleClassifier(X,Y,w)

        # print('theta=',theta,' j=',j)
        # plt.subplot()
        # plot_(X, Y, j, theta, 'Simple weak linear classifier')
        # plt.show()

        for i in range(numSamples):
            I[i]=getI(X[i,j],Y[i],theta)

        errorrate=I.sum()/numSamples
        # print('errorrate=',errorrate)
        alphaK[m]=np.log(1/errorrate-1)
        sum=0
        for i in range(numSamples):
            # I[i]=getI(X[i,j],Y[i],theta)
            if I[i]:
                t=1
            else:
                t=0
            w[i]=w[i]*np.exp(alphaK[m]*t)
            sum=sum+w[i]
        w=w/sum
        # print('w=',w)
        para[m,0]=j
        para[m,1]=theta
    print('alphaK=',alphaK)
    return alphaK, para

def getI(x,y,theta):
    if x>theta:
        h=1
    else:
        h=-1

    if h==y:
        return 0
    else:
        return 1
