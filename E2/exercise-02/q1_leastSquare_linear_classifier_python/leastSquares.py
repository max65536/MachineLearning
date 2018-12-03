import numpy as np

def leastSquares(data, label):
    # Sum of squared error shoud be minimized
    #
    # INPUT:
    # data        : Training inputs  (num_samples x dim)
    # label       : Training targets (num_samples x 1)
    #
    # OUTPUT:
    # weights     : weights   (dim x 1)
    # bias        : bias term (scalar)

    #####Insert your code here for subtask 1a#####
    # Extend each datapoint x as [1, x]
    # (Trick to avoid modeling the bias term explicitly
    
    data = np.array(data)
    label = np.array(label)
    [N, D] = data.shape
    data_new = np.zeros((N, D + 1))
    weight_all = np.zeros(D+1)
    for n in range(N):
        for d in range(D + 1):
            if d == 0:
                data_new[n][d] = 1
            else:
                data_new[n][d] = data[n][d-1]

    array_temp = np.zeros((D+1, D+1))
    for d1 in range(D+1):
        for d2 in range(D+1):
            temp = 0
            for n in range(N):
                temp = temp + data_new[n][d1] * data_new[n][d2]
            array_temp[d1][d2] = temp

    array_temp = np.linalg.inv(array_temp)
    array_temp1 = np.zeros((D+1, N))
    for d1 in range(D+1):
        for n in range(N):
            temp = 0
            for d2 in range (D+1):
                temp = temp + array_temp[d1][d2] * data_new[n][d2]
            array_temp1[d1][n] = temp

    for d1 in range(D+1):
        temp = 0
        for n in range(N):
            temp = temp + array_temp1[d1][n] * label[n]
        weight_all[d1] = temp

    weight = weight_all[1:]
    bias = weight_all[0]
    return weight, bias
