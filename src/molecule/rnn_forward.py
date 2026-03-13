import numpy as np
import src.shared.activation_function as af
import src.atom.rnn_cell_forward as rnn_cell_forward

def rnn_forward(x, a0, parameters, act_h, act_y, rnn_type):

    """
    Implement the molecule of RNN over an entire sequence of data

    Arguments:
        x: imput data for every time step, shape (n_x, m, T_x)
        a0: initial hidden state, shape (n_a,  m)
        parameters: dictionary containing Wax, Waa, Wya, ba, by
        act_h: string, hidden state activation - "tanh", "relu", "sigmoid"
        act_y: string, output layer activation - "sigmoid", "softmax"
        rnn_type : string, rnn cell type - "standard", "GRU"
    Returns:
        y_pred: prediction for every timestep, shape (n_y, m, T_x)
        a: hidden state for every timestep, shape (n_a, m, T_x)
        caches : caches containing parameter at all timestamp
    """

    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape

    y_pred = np.zeros((n_y, m, T_x))
    a = np.zeros((n_a, m, T_x))

    caches = []

    a_next = a0

    if rnn_type == "standard":
        cell = rnn_cell_forward.StandardRNNCell(parameters, act_h, act_y)
    elif rnn_type == "GRU":
        cell = rnn_cell_forward.GRUCell(parameters, act_h, act_y)
    else:
        raise ValueError("Unsupported RNN type - should use standard or GRU")

    for t in range(T_x):
        xt = x[:, :, t]

        a_next, y_pred_t, cache = cell.forward(xt, a_next)

        a[:, :, t] = a_next
        y_pred[:, :, t] = y_pred_t

        caches.append(cache)

    return y_pred, a, caches