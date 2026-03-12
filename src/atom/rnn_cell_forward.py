import numpy as np
import shared.activation_function as af

def rnn_cell_forward (xt, a_prev, parameters, act_h, act_y):
    """
    Implements a single atom of the RNN cell 
    
    Arguments:
        xt: input data at timestamp t, shape (n_x, m)
        a_prev: hidden state at timestamp t-1, shape (n_a, m)
        paramters:
            Wax: weight matrix multiplying the input, shape (n_a, n_x)
            Waa: weight matrix multiplying the hidden state, shape (n_a, n_a)
            Wya: weight matrix relating the hidden state to the output, shape (n_y, n_a)
            ba: Bias relating the hidden state, shape (n_a, m)
            by: Bias relating the output, shape (n_y, 1)
        act_h: string, hidden state activation - "tanh", "relu", "sigmoid"
        act_y: string, output layer activation - "sigmoid", "softmax"

    Returns:
        a_next: next hidden state at timestamp t, shape (n_a, m)
        yt_pred: prediction at timestamp t, shape (n_y, m)
    """

    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    a_next = np.dot(Waa, a_prev)+ np.dot(Wax, xt) + ba

    if act_h == "tanh":
        a_next = af.tanh(a_next)
    elif act_h == "relu":
        a_next = af.relu(a_next)
    elif act_h == "sigmoid":
        a_next = af.sigmoid(a_next)
    else:
        raise ValueError("Unsupported activation function for hidden state %s. Should use tanh or relu or sigmoid", act_h)

    yt_pred = np.dot(Wya, a_next) + by

    if act_y == "sigmoid":
        yt_pred = af.sigmoid(yt_pred)
    elif act_y == "softmax":
        yt_pred = af.softmax(yt_pred)
    else:
        raise ValueError("Unsupported activation function for output layer %s. Should use sigmoid or softmax", act_y)

    return a_next, yt_pred