import numpy as np


def eval_adaBoost_simpleClassifier(X, alphaK, para):
    # INPUT:
    # para	: parameters of simple classifier (K x 2) - for each round
    #           : dimension 1 is j
    #           : dimension 2 is theta
    # K         : number of classifiers used
    # X         : test data points (numSamples x numDim)
    #
    # OUTPUT:
    # classLabels: labels for data points (numSamples x 1)
    # result     : weighted sum of all the K classifier (numSamples x 1)

    #####Insert your code here for subtask 1c#####
    numSamples=len(X)
    K=len(para)
    j=para[:,0].astype(np.int32)

    theta=para[:,1]
    print('j=',j)
    print('theta=',theta)

    result=np.zeros(numSamples)
    classLabels=np.zeros(numSamples,dtype=int)
    for i in range(numSamples):
        voteA=0
        voteB=0
        for m in range(K):
            res=functionC(X[i,j[m]],theta[m])
            if res==1:
                voteA=voteA+alphaK[m]
            else:
                voteB=voteB+alphaK[m]
        if voteA>voteB:
            # result[i]=voteA
            classLabels[i]=1
        else:
            # result[i]=voteB
            classLabels[i]=-1
        result[i]=voteA-voteB
    # print('result=',result)
    # result=alphaK
    return classLabels, result

def functionC(x,theta):
    if (x>theta):
        return 1
    else:
        return -1
