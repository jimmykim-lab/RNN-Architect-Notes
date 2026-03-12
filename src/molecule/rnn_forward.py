import numpy as np
import shared.activation_function as af
import atom.rnn_cell_forward as rnn_cell_forward

def rnn_forward(x, a0, parameters, act_h, act_y):

    """
    Implement the molecule of a simple RNN over an entire sequence of data

    Arguments:
        x: imput data for every time step, shape (n_x, m, T_x)
        a0: initial hidden state, shape (n_a,  m)
        parameters: dictionary containing Wax, Waa, Wya, ba, by
        act_h: string, hidden state activation - "tanh", "relu", "sigmoid"
        act_y: string, output layer activation - "sigmoid", "softmax"
    Returns:
        y_pred: prediction for every timestep, shape (n_y, m, T_x)
        a: hidden state for every timestep, shape (n_a, m, T_x)
    """

    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape

    y_pred = np.zeros((n_y, m, T_x))
    a = np.zeros((n_a, m, T_x))

    a_next = a0

    for t in range(T_x):
        xt = x[:, :, t]

        a_next, y_pred_t = rnn_cell_forward.rnn_cell_forward(xt, a_next, parameters, act_h, act_y)

        a[:, :, t] = a_next
        y_pred[:, :, t] = y_pred_t

    return y_pred, a