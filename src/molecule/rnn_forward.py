import numpy as np
import src.shared.activation_function as af
import src.atom.rnn_cell_forward as rnn_cell_forward

def rnn_forward(x, a0, c0, parameters, act_h, act_y, rnn_type):

    """
    Implement the molecule of RNN over an entire sequence of data

    Arguments:
        x: imput data for every time step, shape (n_x, m, T_x)
        a0: initial hidden state, shape (n_a,  m)
        c0: initial cell state, shape (n_c, m) - only for LSTM
        parameters: dictionary containing weights and biases
        act_h: string, hidden state activation - "tanh", "relu", "sigmoid"
        act_y: string, output layer activation - "sigmoid", "softmax"
        rnn_type : string, rnn cell type - "standard", "GRU"
    Returns:
        y_pred: prediction for every timestep, shape (n_y, m, T_x)
        a: hidden state for every timestep, shape (n_a, m, T_x)
        c: next cell state for every timestep, shape (n_c, m, T_x) - only for LSTM
        caches : caches containing parameter at all timestamp
    """

    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape

    y_pred = np.zeros((n_y, m, T_x))
    a = np.zeros((n_a, m, T_x))

    c = None
    if rnn_type == "LSTM":
        n_c = parameters["Wca"].shape[0]
        c = np.zeros((n_c, m, T_x))

    caches = []

    a_next = a0
    if rnn_type == "LSTM":
        c_next = c0 if c0 is not None else np.zeros((n_c, m))

    if rnn_type == "standard":
        cell = rnn_cell_forward.StandardRNNCell(parameters, act_h, act_y)
    elif rnn_type == "GRU":
        cell = rnn_cell_forward.GRUCell(parameters, act_h, act_y)
    elif rnn_type == "LSTM":
        cell = rnn_cell_forward.LSTMCell(parameters, act_h, act_y)
    else:
        raise ValueError("Unsupported RNN type - should use standard or GRU or LSTM")

    for t in range(T_x):
        xt = x[:, :, t]

        if rnn_type == "LSTM":
            a_next, c_t, y_pred_t, cache = cell.forward(xt, a_next, c_next)
            c[:, :, t] = c_t
            c_next = c_t
        else:
            a_next, y_pred_t, cache = cell.forward(xt, a_next)

        a[:, :, t] = a_next
        y_pred[:, :, t] = y_pred_t

        caches.append(cache)

    return y_pred, a, c, caches