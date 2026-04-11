import numpy as np
import src.shared.activation_function as af

class ScaledDotProductAttention:
    def __init__(self, parameters):
        """
        Initialize a Scaled Dot-Product Attention

        parameters:
            WQ: weight matrix to calculate Q, shape (d_model, d_k)
            WK: weight matrix to calculate K, shape (d_model, d_k)
            WV: weight matrix to calculate V, shape (d_model, d_k)
        """

        self.WQ = parameters["WQ"]
        self.WK = parameters["WK"]
        self.WV = parameters["WV"]


    def forward(self, X):
        """
        Implements a Scaled Dot-Product Attention
        
        Arguments:
            X: input data, shape (T, d_model) ; T = sequence length, d_model = embedding dimension

        Returns:
            attention: output data, shape (T, d_k)
        """

        Q = np.dot(X, self.WQ)
        K = np.dot(X, self.WK)
        V = np.dot(X, self.WV)

        d_k = self.WQ.shape[1]

        attention = np.dot(af.softmax_row(np.dot(Q, K.T) / np.sqrt(d_k)), V) 

        return attention
    
class MultiHeadAttention:
    def __init__(self, parameters, h):
        """
        Initialize a Multi-Head Attention
        
        parameters:
            h: number of attention heads
            WO: weight matrix to project concatenated heads, shape (h * d_k, d_model)
            parameters["head_i"]: dict containing WQ, WK, WV for each head i
        """

        self.heads = [
            ScaledDotProductAttention(parameters[f"head_{i}"])
            for i in range(h)
        ]

        self.WO = parameters["WO"] 
    
    def forward(self, X):
        """
        Implements a Multi-Head Attention
        
        Arguments:
            X: input data, shape (T, d_model) ; T = sequence length, d_model = embedding dimension

        Returns:
            output: output data, shape (T, d_model)
        """

        head_outputs = [head.forward(X) for head in self.heads]
        concat = np.concatenate(head_outputs, axis=-1)
        output = np.dot(concat, self.WO)

        return output
    
class AddNorm:
    def __init__(self, parameters):
        """
        Initialize a AddNorm layer
        
        parameters:
            gamma, beta: learnable parameters, shape (d_model,)
        """
        self.gamma = parameters["gamma"]
        self.beta = parameters["beta"]

    def forward(self, X, sublayer_output):
        """
        Implements a AddNorm layer
        
        Arguments:
            X: input data, shape (T, d_model) ; T = sequence length, d_model = embedding dimension
            sublayer_output: sublayer output, shape (T, d_model)
            
        Returns:
            output: output data after residual and normalization, shape (T, d_model)
        """

        residual = X + sublayer_output

        mean = np.mean(residual, axis=-1, keepdims=True)
        var = np.var(residual, axis=-1, keepdims=True)
        eps = 1e-6

        normalized = (residual - mean) / np.sqrt(var + eps)
        output = self.gamma * normalized + self.beta 

        return output

class FFN:
    def __init__(self, parameters):
        """
        Initialize a FFN

        parameters:
            W1: weight matrix, shape (d_model, d_ff)
            W2: weight matrix, shape (d_ff, d_model)
            b1: bias, shape (d_ff,)
            b2: bias, shape (d_model,)
        """

        self.W1 = parameters["W1"]
        self.W2 = parameters["W2"]
        self.b1 = parameters["b1"]
        self.b2 = parameters["b2"]
    
    def forward(self, X):
        """
        Implements a FFN

        Arguments:
            X: input data, shape (T, d_model)

        Returns:
            output: output data, shape (T, d_model)
        """
        linear = np.dot(X, self.W1) + self.b1
        relu = af.relu(linear)
        output = np.dot(relu, self.W2) + self.b2

        return output
    
class TransformerEncoderLayer:
    def __init__(self, parameters, h):
        """
        Initialize a Transformer Encoder Layer
        
        parameters:
            h: number of attention heads
            mha_parameters: dict containing MultiHeadAttention parameters
            addnorm_1_parameters: dict containing AddNorm parameters (after MHA)
            ffn_parameters: dict containing FFN parameters
            addnorm_2_parameters: dict containing AddNorm parameters (after FFN)
        """
        self.mha = MultiHeadAttention(parameters["mha"], h)
        self.addnorm_1 = AddNorm(parameters["addnorm_1"])
        self.ffn = FFN(parameters["ffn"])
        self.addnorm_2 = AddNorm(parameters["addnorm_2"])

    def forward(self, X):
        """
        Implements a Transformer Encoder Layer

        Arguments:
            X: input data, shape (T, d_model)

        Returns:
            output: output data, shape (T, d_model)
        """
        mha_output = self.mha.forward(X)
        addnorm_1_output = self.addnorm_1.forward(X, mha_output)
        ffn_output = self.ffn.forward(addnorm_1_output)
        output = self.addnorm_2.forward(addnorm_1_output, ffn_output)

        return output

