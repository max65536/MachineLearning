def linclass(weight, bias, data):
    # Linear Classifier
    #
    # INPUT:
    # weight      : weights                (dim x 1)
    # bias        : bias term              (scalar)
    # data        : Input to be classified (num_samples x dim)
    #
    # OUTPUT:
    # class_pred       : Predicted class (+-1) values  (num_samples x 1)

    #####Insert your code here for subtask 1b#####
    # Perform linear classification i.e. class prediction
    weight = np.array(weight)
    dim = len(weight)
    N = len(data)
    class_pred = np.zeros(N)
    for n in range(N):
        temp = 0
        for d in range(dim):
            temp = temp + weight[d] * data[n][d]
        temp = temp + bias
        if (temp > 0):
            class_pred[n] = 1
        else:
            class_pred[n] = -1
    return class_pred


