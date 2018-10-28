import numpy as np
from getLogLikelihood import getLogLikelihood
def MStep(gamma, X):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).

    #####Insert your code here for subtask 6c#####
    N=len(X)
    K=len(gamma[0])
    D=len(X[0])
    Nh=gamma.sum(axis=0)
    weights=Nh/N

    means=np.dot(gamma.T,X)/Nh[:,None]
    # for j in range(K):
    #     tmp=np.zeros(D)
    #     for n in range(N):
    #         tmp=tmp+gamma[j,n]*X[n]



    covariances=np.zeros((D,D,K))
    for k in range(0,K):

        tmp=np.zeros((D,D))
        for i in range(0,N):
            sub=X[i]-means[k]
            tmp=tmp+gamma[i,k]*np.outer(sub,sub)
        covariances[:,:,k]=tmp/Nh[k]

    logLikelihood=getLogLikelihood(means, weights, covariances, X)
    # logLikelihood=0

    return [weights, means, covariances, logLikelihood]

def test_data():
    gamma=np.array(
        [[10,5],
        [45,4],
        [35,41],
        [12,65],
        [12,36]
                ])
    X=np.array(
        [[10,7,5],
        [114,45,4],
        [345,65,41],
        [112,96,65],
        [12,98,36]
                ])
    Nh=gamma.sum(axis=0)/5
    # print('Nh=',Nh)
    tmp=np.dot(gamma.T,X)
    # print(tmp)
    means=tmp/Nh[:,None]
    # print("means=",means)

    D=3
    K=2
    N=5
    weights=Nh/N
    covariances=np.zeros((D,D,K))
    for k in range(0,K):
        sub=X-means[k]
        tmp=np.zeros((D,D))
        for i in range(0,N):
            tmp=gamma[i,k]*np.outer(sub[i],sub[i])
        covariances[:,:,k]=tmp
    # for k in range(0,K):
    #     print("covariances=",covariances[:,:,k])
    print(covariances)
    return [means,weights,covariances,X]
# test_data()
