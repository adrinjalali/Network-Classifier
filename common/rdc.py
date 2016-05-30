import numpy as np
import scipy.stats
import sklearn.cross_decomposition

def rdc(x, y, s=1/6, k=20, f=np.sin):
    x_ = np.vstack((scipy.stats.rankdata(x) / len(x), np.ones(len(x)))).transpose()
    tx = np.array(np.random.normal(size = x_.shape[1] * k))
    tx = np.reshape(tx, (2,-1))
    x__ = np.dot(s * x_, tx)
    
    y_ = np.vstack((scipy.stats.rankdata(y) / len(y), np.ones(len(y)))).transpose()
    ty = np.array(np.random.normal(size = y_.shape[1] * k))
    ty = np.reshape(ty, (2,-1))
    y__ = np.dot(s * y_, ty)
    
    x___ = np.hstack((f(x__), np.ones(shape=(x__.shape[0], 1))))
    y___ = np.hstack((f(y__), np.ones(shape=(y__.shape[0], 1))))
    
    cca = sklearn.cross_decomposition.CCA(n_components=1)
    U_c, V_c = cca.fit_transform(x___, y___)
    
    result = np.corrcoef(U_c.T, V_c.T)[0,1]
    return result

