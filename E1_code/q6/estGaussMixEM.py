import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from EStep import EStep
from MStep import MStep
from regularize_cov import regularize_cov


def estGaussMixEM(data, K, n_iters, epsilon):
    # EM algorithm for estimation gaussian mixture mode
    #
    # INPUT:
    # data           : input data, N observations, D dimensional
    # K              : number of mixture components (modes)
    #
    # OUTPUT:
    # weights        : mixture weights - P(j) from lecture
    # means          : means of gaussians
    # covariances    : covariancesariance matrices of gaussians
    # logLikelihood  : log-likelihood of the data given the model

    #####Insert your code here for subtask 6e#####

    X=np.array(data)
    N=len(X)
    D=len(X[0])
    weights=np.ones(K)
    [means,covariances] = K_Means(data, K,D)
    # for i in range(K):
    #     covariances[:,:,i]=regularize_cov(covariances[:,:,i], epsilon)
    for n in range(n_iters):
        [logLikelihood,gamma]=EStep(means, covariances, weights, X)
        [weights,means,covariances,logLikelihood]=MStep(gamma, X)
        # for i in range(K):
        #     covariances[:,:,i]=regularize_cov(covariances[:,:,i], epsilon)

    return [weights, means, covariances]

def K_Means(data,K,D):
    n_dim=D
    kmeans = KMeans(n_clusters = K, n_init=10).fit(data)
    cluster_idx = kmeans.labels_
    means = kmeans.cluster_centers_

    # Create initial covariance matrices
    covariances=np.zeros((D,D,K))
    for j in range(K):
        data_cluster = data[cluster_idx == j]
        min_dist = np.inf
        for i in range(K):
            # compute sum of distances in cluster
            dist = np.mean(euclidean_distances(data_cluster, [means[i]], squared=True))
            if dist < min_dist:
                min_dist = dist
        covariances[:, :, j] = np.eye(n_dim) * min_dist
    return [means,covariances]

# print(np.ones(5))
