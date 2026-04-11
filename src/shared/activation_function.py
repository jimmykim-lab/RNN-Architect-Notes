import numpy as np

def softmax (z):
    """
    Computes the softmax of vector z, axis=0, column-wise normalization (for RNN output)

    Arguments:
        z: input array, shape (n_y, m)
    
    Returns:
        s: softmax output array, shape (n_y, m)
    """

    e_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    s = e_z / np.sum(e_z, axis=0, keepdims=True) 

    return s

def softmax_row(z):
    """
    Computes the softmax of vector z, axis=-1, row-wise normalization (for Transformer attention)

    Arguments:
        z: input array, shape (n_y, m)
    
    Returns:
        s: softmax output array, shape (n_y, m)
    """

    e_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    s = e_z / np.sum(e_z, axis=-1, keepdims=True)

    return s


def sigmoid (z):
    """
    Computes the sigmoid of vector z

    Arguments:
        z: input array, shape (n_y, m)
    
    Returns:
        s: sigmoid output array, shape (n_y, m)
    """

    s = 1 / (1 + np.exp(-z))

    return s

def relu (z):
    """
    Computes the relu of vector z

    Arguments:
        z: input array, shape (n_y, m)
    
    Returns:
        s: relu output array, shape (n_y, m)
    """

    s = np.maximum(0, z)

    return s

def tanh(z):
    """
    Computes the tanh of vector z

    Arguments:
        z: input array, shape (n_y, m)
    
    Returns:
        s: tanh output array, shape (n_y, m)
    """

    s = np.tanh(z)

    return s
