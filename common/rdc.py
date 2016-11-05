import numpy as np
import scipy.stats
import sklearn.cross_decomposition
from numpy.linalg import lstsq,eig
from numpy import cov,dot,arange,c_

def cca(x_tn,y_tm, reg=0.00000001):
    x_tn = x_tn-x_tn.mean(axis=0)
    y_tm = y_tm-y_tm.mean(axis=0)
    N = x_tn.shape[1]
    M = y_tm.shape[1]
    xy_tq = c_[x_tn,y_tm]
    cqq = cov(xy_tq,rowvar=0)
    cxx = cqq[:N,:N]+reg*np.eye(N)+0.000000001*np.ones((N,N))
    cxy = cqq[:N,N:(N+M)]+0.000000001*np.ones((N,N))
    cyx = cqq[N:(N+M),:N]+0.000000001*np.ones((N,N))
    cyy = cqq[N:(N+M),N:(N+M)]+reg*np.eye(N)+0.000000001*np.ones((N,N))
    
    K = min(N,M)
    
    xldivy = lstsq(cxx,cxy)[0]
    yldivx = lstsq(cyy,cyx)[0]
    #print xldivy
    #print dot(np.linalg.inv(cxx),cxy)
    _,vecs = eig(dot(xldivy,yldivx))
    a_nk = vecs[:,:K]
    #print normr(vecs.T)
    b_mk = dot(yldivx,a_nk)

    u_tk = dot(x_tn,a_nk)
    v_tk = dot(y_tm,b_mk)

    return a_nk,b_mk,u_tk,v_tk
                                                                                        

def rdc(x, y, s=1/6, k=20, f=np.sin):
    x_ = np.vstack((scipy.stats.rankdata(x) / len(x), np.ones(len(x)))).transpose()
    tx = np.array(np.random.normal(size = x_.shape[1] * k))
    tx = np.reshape(tx, (2,-1))
    x_ = np.dot(s * x_, tx)
    
    y_ = np.vstack((scipy.stats.rankdata(y) / len(y), np.ones(len(y)))).transpose()
    ty = np.array(np.random.normal(size = y_.shape[1] * k))
    ty = np.reshape(ty, (2,-1))
    y_ = np.dot(s * y_, ty)
    
    x_ = np.hstack((f(x_), np.ones(shape=(x_.shape[0], 1))))
    y_ = np.hstack((f(y_), np.ones(shape=(y_.shape[0], 1))))
    
    #cca = sklearn.cross_decomposition.CCA(n_components=1)
    #U_c, V_c = cca.fit_transform(x_, y_)
    a, b, U_c, V_c = cca(x_, y_)
    
    result = np.corrcoef(U_c, V_c)[0,1].real
    return result
