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
            raise ValueError(f"Unsupported activation function for hidden state {self.act_h}. Should use tanh or relu or sigmoid")

        yt_pred = np.dot(self.Wya, a_next) + self.by

        if self.act_y == "sigmoid":
            yt_pred = af.sigmoid(yt_pred)
        elif self.act_y == "softmax":
            yt_pred = af.softmax(yt_pred)
        else:
            raise ValueError(f"Unsupported activation function for output layer {self.act_h}. Should use sigmoid or softmax")
        
        cache = (a_next, a_prev, xt, self)

        return a_next, yt_pred, cache
    
class GRUCell:
    def __init__(self, parameters, act_h, act_y):
        """
        Initialize the GRU cell with optimized matrix concatenation.
        This approach reduces the number of expensive np.dot operations.

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
        self.Wux = parameters["Wux"]
        self.Wua = parameters["Wua"]
        self.Wrx = parameters["Wrx"]
        self.Wra = parameters["Wra"]
        self.Wya = parameters["Wya"]

        # [Optimization] Horizontally stack weights to handle a_prev and xt in one shot
        self.Wa = np.hstack((parameters["Waa"],parameters["Wax"]))
        self.Wu = np.hstack((parameters["Wua"],parameters["Wux"]))
        self.Wr = np.hstack((parameters["Wra"],parameters["Wrx"]))

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

        # [Optimization] Vertically stack previous state and current input
        gate_input = np.vstack((a_prev, xt))

        # [Update Gate] : Decides how much of the previous memory to carry over to the future.
        #update_gate = af.sigmoid(np.dot(self.Wua, a_prev) + np.dot(self.Wux, xt) + self.bu)
        update_gate = af.sigmoid(np.dot(self.Wu, gate_input) + self.bu)

        # [Reset Gate] : Decides how much of the past state to ignore when calculating the candidate.
        #reset_gate = af.sigmoid(np.dot(self.Wra, a_prev) + np.dot(self.Wrx, xt) + self.br)
        reset_gate = af.sigmoid(np.dot(self.Wr, gate_input) + self.br)

        # [Candidate hidden state] : Represents new information to be potentially added to the state
        candidate_input = np.vstack((reset_gate * a_prev, xt))
        candidate = np.dot(self.Wa, candidate_input) + self.ba

        if self.act_h == "tanh":
            candidate = af.tanh(candidate)
        elif self.act_h == "relu":
            candidate = af.relu(candidate)
        elif self.act_h == "sigmoid":
            candidate = af.sigmoid(candidate)
        else:
            raise ValueError(f"Unsupported activation function for hidden state {self.act_h}. Should use tanh or relu or sigmoid")

        a_next = update_gate * candidate + (1 - update_gate) * a_prev

        yt_pred = np.dot(self.Wya, a_next) + self.by

        if self.act_y == "sigmoid":
            yt_pred = af.sigmoid(yt_pred)
        elif self.act_y == "softmax":
            yt_pred = af.softmax(yt_pred)
        else:
            raise ValueError(f"Unsupported activation function for output layer {self.act_y}. Should use sigmoid or softmax")

        cache = (a_next, candidate, update_gate, reset_gate, xt, a_prev, self)

        return a_next, yt_pred, cache
    
class LSTMCell:
    def __init__(self, parameters, act_h, act_y):
        """
        Initialize the LSTM cell with optimized matrix concatenation.
        This approach reduces the number of expensive np.dot operations.

        paramters:
            Wca: weight matrix multiplying the previous hidden state calculating candidate hidden state, shape (n_c, n_a)
            Wcx: weight matrix multiplying the input calculating candidate hidden state, shape (n_c, n_x)
            Wua: weight matrix multiplying the previous hidden state calculating the update gate, shape (n_u, n_a)
            Wux: weight matrix multiplying the input calculating the update gate, shape (n_u, n_x)
            Wfa: weight matrix multiplying the previous hidden state calculating the forget gate, shape (n_f, n_a)
            Wfx: weight matrix multiplying the input calculating the forget gate, shape (n_f, n_x)
            Woa: weight matrix multiplying the previous hidden state calculating the output gate, shape (n_o, n_a)
            Wox: weight matrix multiplying the input calculating the output gate, shape (n_o, n_x)
            Wya: weight matrix relating the hidden state to the output, shape (n_y, n_a)

            bc: bias relating the candidate hidden state, shape (n_c, m)
            bu: bias relating the update gate, shape (n_u, m)
            bf: bias relating the forget gate, shape (n_f, m)
            bo: bias relating the output gate, shape (n_o, m)
            by: bias relating the output, shape (n_y, 1)

            act_h: string, hidden state activation - "tanh", "relu", "sigmoid"
            act_y: string, output layer activation - "sigmoid", "softmax"
        """
        self.Wca = parameters["Wca"]
        self.Wcx = parameters["Wcx"]
        self.Wua = parameters["Wua"]
        self.Wux = parameters["Wux"]
        self.Wfa = parameters["Wfa"]
        self.Wfx = parameters["Wfx"]
        self.Woa = parameters["Woa"]
        self.Wox = parameters["Wox"]
        self.Wya = parameters["Wya"]

        # [Optimization] Horizontally stack weights to handle a_prev and xt in one shot
        self.Wc = np.hstack((parameters["Wca"],parameters["Wcx"]))
        self.Wu = np.hstack((parameters["Wua"],parameters["Wux"]))
        self.Wf = np.hstack((parameters["Wfa"],parameters["Wfx"]))
        self.Wo = np.hstack((parameters["Woa"],parameters["Wox"]))

        self.bc = parameters["bc"]
        self.bu = parameters["bu"]
        self.bf = parameters["bf"]
        self.bo = parameters["bo"]
        self.by = parameters["by"]

        self.act_h = act_h
        self.act_y = act_y

    def forward (self, xt, a_prev, c_prev):
        """
        Implements a single atom of the LSTM cell

        Arguments:
            xt: input data at timestamp t, shape (n_x, m)
            a_prev: hidden state at timestamp t-1, shape (n_a, m)
            c_prev: cell state at timestamp t-1, shape (n_c, m)

        Returns:
            a_next: next hidden state at timestamp t, shape (n_a, m)
            c_next: next cell state at timestamp t, shape (n_c, m)
            yt_pred: prediction at timestamp t, shape (n_y, m)
            cache : cache containing a_next, c_next, update_gate, forget_gate, output_gate, xt, a_prev, c_prev, self(hyper parameter) at timestamp t    
        """

        # [Optimization] Vertically stack previous state and current input
        gate_input = np.vstack((a_prev, xt))

        # [Update Gate] : Decides how much of the current candidate to ignore when calculating the next candidate.
        u_gate = af.sigmoid(np.dot(self.Wu, gate_input) + self.bu)

        # [Forget Gate] : Decides how much of the past candidate to ignore when calculating the next candidate.
        f_gate = af.sigmoid(np.dot(self.Wf, gate_input) + self.bf)

        # [Output Gate] : Decides how much of the next candidate to ignore when calculating the next hidden state.
        o_gate = af.sigmoid(np.dot(self.Wo, gate_input) + self.bo)

        # [Candidate cell state] : Represents new information to be potentially added to the next cell state
        c_tilde = np.dot(self.Wc, gate_input) + self.bc
        if self.act_h == "tanh":
            c_tilde = af.tanh(c_tilde)
        elif self.act_h == "relu":
            c_tilde = af.relu(c_tilde)
        elif self.act_h == "sigmoid":
            c_tilde = af.sigmoid(c_tilde)
        else:
            raise ValueError(f"Unsupported activation function for hidden state {self.act_h}. Should use tanh or relu or sigmoid" )

        # [Cell state] : Represents new information to be the next cell state
        c_next = (u_gate * c_tilde) + (f_gate * c_prev)

        # [Nexe hidden state] : Represents new information to the next hidden state
        if self.act_h == "tanh":
            a_next = af.tanh(c_next)
        elif self.act_h == "relu":
            a_next = af.relu(c_next)
        elif self.act_h == "sigmoid":
            a_next = af.sigmoid(c_next)
        else:
            raise ValueError(f"Unsupported activation function for hidden state {self.act_h}. Should use tanh or relu or sigmoid")

        a_next = o_gate * a_next

        # [Prediction] : Represents new information to the prediction
        yt_pred = np.dot(self.Wya, a_next) + self.by

        if self.act_y == "sigmoid":
            yt_pred = af.sigmoid(yt_pred)
        elif self.act_y == "softmax":
            yt_pred = af.softmax(yt_pred)
        else:
            raise ValueError(f"Unsupported activation function for output layer {self.act_y}. Should use sigmoid or softmax")

        cache = (a_next, c_next, u_gate, f_gate, o_gate, xt, a_prev, c_prev, self)

        return a_next, c_next, yt_pred, cache