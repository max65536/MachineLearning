import numpy as np
from kern import kern
import cvxopt as cvx

def svmkern(X, t, C, p):
    # Non-Linear SVM Classifier
    #
    # INPUT:
    # X             : the dataset                        (dim x num_samples)
    # t             : labeling                           (1 x num_samples)
    # C             : penalty factor the slack variables (scalar)
    # p             : order of the polynom               (scalar)
    #
    # OUTPUT:
    # sv            : support vectors (boolean)          (1 x num_samples)
    # b             : bias of the classifier             (scalar)
    # slack         : points inside the margin (boolean) (1 x num_samples)

    #####Insert your code here for subtask 2d#####
    # we want to get alpha_n >= 0 for all n : positive Lagrange multipliers
    # maximize svm dual formulation, which is quadratic programming problem
    # X=X.T
    dim=len(X)
    num_samples=len(X[0])
    # print('X=',X)
    # print('t=',t)
    # print('C=',C)
    # print('num_samples',num_samples)
    # print('dim=',dim)

    H=np.zeros((num_samples,num_samples))
    # print('H.shape1=',H.shape[1])
    Xi=np.hsplit(X, num_samples)
    for i in range(num_samples):
        for j in range(num_samples):
            H[i,j]=t[i]*t[j]*kern(Xi[i],Xi[j],p)[0,0]

    H=cvx.matrix(H)
    q=cvx.matrix((-1)*np.ones(num_samples))
    # G=cvx.matrix(np.vstack([-np.eye(num_samples,dtype=float),np.eye(num_samples)]))
    # G=cvx.matrix(-np.eye(num_samples,dtype=float))
    G=cvx.matrix(np.vstack([-np.eye(num_samples),np.eye(num_samples)]))
    # h=cvx.matrix([0.,C])
    h=cvx.matrix(np.hstack([np.zeros(num_samples),C*np.ones(num_samples)]))
    A=cvx.matrix(t.reshape(1,num_samples))
    # A=cvx.matrix(1.0, (1,num_samples))
    b0=cvx.matrix(0.)
    # print('H=',H)
    # print('q=',q)
    # print('G=',G)
    # print('h=',h)
    # print('A=',A)
    # print('b0=',b0)

    solution=cvx.solvers.qp(H,q,G,h,A,b0)
    alpha=solution['x']
    # print('alpha=',alpha)

    w=np.zeros(dim)
    for i in range(num_samples):
        for j in range(dim):
            w[j]=w[j]+alpha[i]*t[i]*X[j,i]


    sv=np.zeros(num_samples,dtype="b")
    b=0
    count=0
    for s in range(num_samples):
        tmp=0
        if alpha[s]<1e-4:
            continue
        sv[s]=True
        count=count+1
        for j in range(num_samples):
            multiplyres=np.dot(Xi[j].T,Xi[s])
            # print('multiplyres=',multiplyres)
            # print('alpha3= ',alpha[3],'t3=',t[3])
            tmp=tmp+alpha[j]*t[j]*multiplyres[0,0]
        b=b+t[s]-tmp
        # print('b=',t[s]-tmp,'  alpha=',alpha[s])
    b=b/count
    # b=1.19782218

    result=np.zeros(num_samples)
    slack=np.zeros(num_samples,dtype="b")

    for i in range(num_samples):
        tmp=kernfunc(Xi[i],Xi,b,p)
        # if tmp>=1:
        #     result[i]=1
        # elif tmp<=-1:
        #     result[i]=-1
        # else:
        #     result[i]=1
        #     slack[i]=1
        # if tmp<1 and tmp>-1:
            # slack[i]=True
        if tmp>=0:
            result[i]=1
        else:
            result[i]=-1

        if abs(tmp)<1:
            sv[i]=True
            slack[i]=True

    # result=t
    # print('w=',w)
    # print('b=',b)
    return alpha, sv, b, result, slack

def kernfunc(x,Xi,b,p):
    res=0
    for i in range(len(x)):
        res=res+kern(x,Xi[i],p)[0,0]
    res=res+b
    # return np.dot(x.T,w)+b
    return res
