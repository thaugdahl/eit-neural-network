import numpy as np

def xy(vect):
    return vect[:,:,0]*vect[:,:,1]

def xz(vect):
    return vect[:,-2:,0]*vect[:,-2:,2]

def inject_constant(row):
    """
    Silly function that returns 1.
    :param row: Not meaningful
    :return: Returns the int 1
    """
    return 1

def cos_t(freq):
    def func(vect):
        return np.cos(freq*vect[:,:,-1])
    return func

def x_cubed(vect):
    return vect[:,:,0]**3