import numpy as np
from getLogLikelihood import getLogLikelihood,multiGauss


def EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.

    #####Insert your code here for subtask 6b#####
    logLikelihood=getLogLikelihood(means, weights, covariances, X)
    K=len(weights)
    N=len(X)
    gamma=np.zeros((N,K))
    for i in range(0,N):
        tmp=0
        for k in range(0,K):
            gamma[i,k]=weights[k]*multiGauss(X[i], means[k], covariances[:,:,k])
            tmp=tmp+gamma[i,k]
        for k in range(0,K):
            gamma[i,k]=gamma[i,k]/tmp
    return [logLikelihood, gamma]
