import numpy as np
import src.shared.activation_function as af

class StandardRNNCell:
    def __init__(self, parameters, act_h, act_y):
        """
        Initialize a single atom of standard RNN cell 

        paramters:
            Wax: weight matrix multiplying the input, shape (n_a, n_x)
            Waa: weight matrix multiplying the hidden state, shape (n_a, n_a)
            Wya: weight matrix relating the hidden state to the output, shape (n_y, n_a)
            ba: Bias relating the hidden state, shape (n_a, m)
            by: Bias relating the output, shape (n_y, 1)
            act_h: string, hidden state activation - "tanh", "relu", "sigmoid"
            act_y: string, output layer activation - "sigmoid", "softmax"
        """

        self.Wax = parameters["Wax"]
        self.Waa = parameters["Waa"]
        self.Wa = np.hstack((parameters["Waa"], parameters["Wax"]))
        self.Wya = parameters["Wya"]

        self.ba = parameters["ba"]
        self.by = parameters["by"]

        self.act_h = act_h
        self.act_y = act_y

    def forward (self, xt, a_prev):
        """
        Implements a single atom of standard RNN cell 
        
        Arguments:
            xt: input data at timestamp t, shape (n_x, m)
            a_prev: hidden state at timestamp t-1, shape (n_a, m)
           

        Returns:
            a_next: next hidden state at timestamp t, shape (n_a, m)
            yt_pred: prediction at timestamp t, shape (n_y, m)
            cache : cache containing a_next, a_prev, xt, self(hyper parameter) at timestamp t
        """

        
        # a_next = np.dot(self.Waa, a_prev)+ np.dot(self.Wax, xt) + self.ba
        a_input = np.vstack((a_prev, xt))
        a_next = np.dot(self.Wa, a_input) + self.ba

        if self.act_h == "tanh":
            a_next = af.tanh(a_next)
        elif self.act_h == "relu":
            a_next = af.relu(a_next)
        elif self.act_h == "sigmoid":
            a_next = af.sigmoid(a_next)
        else:
            raise ValueError("Unsupported activation function for hidden state %s. Should use tanh or relu or sigmoid", self.act_h)

        yt_pred = np.dot(self.Wya, a_next) + self.by

        if self.act_y == "sigmoid":
            yt_pred = af.sigmoid(yt_pred)
        elif self.act_y == "softmax":
            yt_pred = af.softmax(yt_pred)
        else:
            raise ValueError("Unsupported activation function for output layer %s. Should use sigmoid or softmax", self.act_y)
        
        cache = (a_next, a_prev, xt, self)

        return a_next, yt_pred, cache
    
class GRUCell:
    def __init__(self, parameters, act_h, act_y):
        """
        Intialize a single atom of the GRU cell

        paramters:
            Wax: weight matrix multiplying the input calculating candidate hidden state, shape (n_a, n_x)
            Waa: weight matrix multiplying the previous hidden state calculating candidate hidden state, shape (n_a, n_a)
            Wux: weight matrix multiplying the input calculating the update gate, shape (n_u, n_x)
            Wua: weight matrix multiplying the previous hidden state calculating the update gate, shape (n_u, n_a)
            Wrx: weight matrix multiplying the input calculating the reset gate, shape (n_r, n_x)
            Wra: weight matrix multiplying the previous hidden state calculating the reset gate, shape (n_r, n_a)
            Wya: weight matrix relating the hidden state to the output, shape (n_y, n_a)
            ba: Bias relating the hidden state, shape (n_a, m)
            bu: Bias relating the update gate, shape (n_u, m)
            br: Bias relating the reset gate, shape (n_r, m)
            by: Bias relating the output, shape (n_y, 1)
            act_h: string, hidden state activation - "tanh", "relu", "sigmoid"
            act_y: string, output layer activation - "sigmoid", "softmax"
        """
        self.Wax = parameters["Wax"]
        self.Waa = parameters["Waa"]
        self.Wa = np.hstack((parameters["Waa"],parameters["Wax"]))
        self.Wux = parameters["Wux"]
        self.Wua = parameters["Wua"]
        self.Wu = np.hstack((parameters["Wua"],parameters["Wux"]))
        self.Wrx = parameters["Wrx"]
        self.Wra = parameters["Wra"]
        self.Wr = np.hstack((parameters["Wra"],parameters["Wrx"]))
        self.Wya = parameters["Wya"]

        self.ba = parameters["ba"]
        self.bu = parameters["bu"]
        self.br = parameters["br"]
        self.by = parameters["by"]

        self.act_h = act_h
        self.act_y = act_y

    def forward (self, xt, a_prev):
        """
        Implements a single atom of the GRU cell

        Arguments:
            xt: input data at timestamp t, shape (n_x, m)
            a_prev: hidden state at timestamp t-1, shape (n_a, m)

        Returns:
            a_next: next hidden state at timestamp t, shape (n_a, m)
            yt_pred: prediction at timestamp t, shape (n_y, m)
            cache : cache containing a_next, candidate, update_gate, reset_gate, xt, a_prev, self(hyper parameter) at timestamp t    
        """

        gate_input = np.vstack((a_prev, xt))

        # Update Gate
        #update_gate = af.sigmoid(np.dot(self.Wua, a_prev) + np.dot(self.Wux, xt) + self.bu)
        update_gate = af.sigmoid(np.dot(self.Wu, gate_input) + self.bu)

        # Reset Gate
        #reset_gate = af.sigmoid(np.dot(self.Wra, a_prev) + np.dot(self.Wrx, xt) + self.br)
        reset_gate = af.sigmoid(np.dot(self.Wr, gate_input) + self.br)

        candidate_input = np.vstack((reset_gate * a_prev, xt))
        # Candidate hidden state
        candidate = np.dot(self.Wa, candidate_input) + self.ba

        if self.act_h == "tanh":
            candidate = af.tanh(candidate)
        elif self.act_h == "relu":
            candidate = af.relu(candidate)
        elif self.act_h == "sigmoid":
            candidate = af.sigmoid(candidate)
        else:
            raise ValueError("Unsupported activation function for hidden state %s. Should use tanh or relu or sigmoid", self.act_h)

        a_next = update_gate * candidate + (1 - update_gate) * a_prev

        yt_pred = np.dot(self.Wya, a_next) + self.by

        if self.act_y == "sigmoid":
            yt_pred = af.sigmoid(yt_pred)
        elif self.act_y == "softmax":
            yt_pred = af.softmax(yt_pred)
        else:
            raise ValueError("Unsupported activation function for output layer %s. Should use sigmoid or softmax", self.act_y)

        cache = (a_next, candidate, update_gate, reset_gate, xt, a_prev, self)

        return a_next, yt_pred, cache