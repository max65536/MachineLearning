import numpy as np
# from numpy import *
from numpy.linalg import *
from math import pi,exp,sqrt
from regularize_cov import regularize_cov
# from MStep import test_data
# import math
def getLogLikelihood(means, weights, covariances, X):
    # Log Likelihood estimation
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each gaussian DxDxK
    # X              : Input data NxD
    # where N is number of data points
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # logLikelihood  : log-likelihood

    #####Insert your code here for subtask 6a#####
    N=len(X)
    D=len(covariances)
    K=len(means)
    P=np.zeros((N,K));
    logLikelihood=0
    # for Xi in X:
    #     for k in range(0,K):
    #         sub=Xi-means[k];
    #         P=1/(pow(2*pi,D/2)*sqrt(np.linalg.det(covariances[:,:,k])))*exp(-0.5*sub.transpose*inv(covariances[:,:,k])*sub)
    #         logLikelihood[k]=logLikelihood+log(P);
    for i in range(0,N):
        tmp=0
        for k in range(0,K):
            P[i,k]=weights[k]*multiGauss(X[i], means[k], covariances[:,:,k])
            tmp=tmp+P[i,k]
        logLikelihood=logLikelihood+np.log(tmp)

    return logLikelihood

def multiGauss(X,mean,covariance):

    # X:1xD
    # mean:1xD
    # covariance:DxD
    # epsilon=0.01
    # covariance=regularize_cov(covariance, epsilon)
    D=len(X)
    sub=X-mean;
    q=1/(pow(2*pi,D/2)*sqrt(np.linalg.det(covariance)))
    # print(np.dot(sub,inv(covariance)))
    # sub=sub/1000000
    # print(sub)
    p=exp(-0.5*np.dot(np.dot(sub,inv(covariance)),sub))
    return q*p

# print(pow(5*pi,3/2))
# X=np.array([[1,2,43,4],
#               [5,55,7,8],
#               [9,109,11,12]])
# print(X.T)

# number=X.size  # 计算 X 中所有元素的个数
# X_row=np.size(X,0)  #计算 X 一行元素的个数
# X_col=np.size(X,1)  #计算 X 一列元素的个数
# for x in X:
#     print(x)

# print("number:",number)
# print("X_row:",X_row)
# print("X_col:",X_col)
# print(inv(X))
# print(X.transpose())
def test():

    [means,weights,covariances,X]=test_data()
    N=5
    D=3
    K=2
    # for i in range(0,K):
    #     sub=X[1]-means[i]
    #     print("sub=",sub)
    #     covariance=covariances[:,:,i]
    #     print("covariance=",covariance)
    #     print("det",np.linalg.det(covariance))
    #     q=1/(pow(2*pi,D/2)*sqrt(10+np.linalg.det(covariance)))
    #     P=exp(-0.5*np.dot(np.dot(sub,inv(covariance)),sub))
    #     print("P=",P)
    P=multiGauss(X[1], means[1], covariances[:,:,1])
    print(P)

# test()
# P=np.zeros(3)
# print(P+0.1)
