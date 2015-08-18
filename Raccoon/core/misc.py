import numpy as np

class Misc:
    def exclude_cols(X, cols):
        """ exludes indices in cols, from columns of X """
        return X[:, ~np.in1d(np.arange(X.shape[1]), cols)]
    
