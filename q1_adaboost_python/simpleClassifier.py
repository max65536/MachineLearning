import numpy as np

def simpleClassifier(X, Y,w):
    # Select a simple classifier
    #
    # INPUT:
    # X         : training examples (numSamples x numDim)
    # Y         : training lables (numSamples x 1)
    #
    # OUTPUT:
    # theta 	: threshold value for the decision (scalar)
    # j 		: the dimension to "look at" (scalar)

    #####Insert your code here for subtask 1b#####
    # print('X=',X)
    # print('Y=',Y)
    numDim=len(X[0])
    numSamples=len(X)
    min=1000000
    j=0
    theta=np.zeros(numDim)
    for j in range(numDim):
        # test theta
        for k in range(numSamples):
            theta_k=X[k,j]
            sum=0
            num=0
            # use n datapoints
            for i in range(numSamples):
                I=getI(X[i,j],Y[i],theta_k)
                sum=sum+I*w[i]
                num=num+I
            # print('sum=',sum,' theta=',theta)
            if min>sum:
                min =sum
                theta[j]=theta_k

    errorrate=num/numSamples
    # print('min=',min, ' errorrate=',num/numSamples)

    min_theta=1000000
    for j in range(numDim):
        if min_theta>theta[j]:
            min_theta=theta[j]
            min_j=j
    # theta=0.5
    return min_j, min_theta

def functionC(x,theta):
    if (x>theta):
        return 1
    else:
        return -1
def getI(x,y,theta):
    if x>theta:
        h=1
    else:
        h=-1

    if h==y:
        return 0
    else:
        return 1
