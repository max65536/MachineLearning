import numpy as np
from estGaussMixEM import estGaussMixEM
from getLogLikelihood import getLogLikelihood,multiGauss


def skinDetection(ndata, sdata, K, n_iter, epsilon, theta, img):
    # Skin Color detector
    #
    # INPUT:
    # ndata         : data for non-skin color
    # sdata         : data for skin-color
    # K             : number of modes
    # n_iter        : number of iterations
    # epsilon       : regularization parameter
    # theta         : threshold
    # img           : input image
    #
    # OUTPUT:
    # result        : Result of the detector for every image pixel

    #####Insert your code here for subtask 1g#####
    nweights, nmeans, ncovariances = estGaussMixEM(ndata, K, n_iter, epsilon)
    sweights, smeans, scovariances = estGaussMixEM(sdata, K, n_iter, epsilon)
    result=np.zeros(img.shape)
    for i in range(len(img)):
        for j in range(len(img[0])):
            nlhd=0
            slhd=0
            for k in range(K):
                nlhd=nlhd+nweights[k]*multiGauss(img[i,j], nmeans[k], ncovariances[:,:,k])
                slhd=slhd+sweights[k]*multiGauss(img[i,j], smeans[k], scovariances[:,:,k])
            # nlhd=getLogLikelihood(nmeans, nweights, ncovariances, img[i,j])
            # slhd=getLogLikelihood(smeans, sweights, scovariances, img[i,j])
            if slhd/nlhd>theta:
                result[i,j]=[255,255,255]

    return result
