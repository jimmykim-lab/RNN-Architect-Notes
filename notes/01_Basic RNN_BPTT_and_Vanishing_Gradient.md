# [Deep Dive] Basic RNN BPTT and the Vanishing Gradient Problem

### **1. Introduction**
In this note, I analyze the mathematical foundations of **Backpropagation Through Time (BPTT)** and investigate why standard RNNs struggle with long-term dependencies. Following my implementation's notation, I use **$a_t$** to represent the activation (hidden state) at time $t$.

---

### **2. Backpropagation Through Time (BPTT)**
RNNs process sequential data by maintaining an activation state $a_t$ that acts as memory.
To train te recurring weight matrix $W_{aa}$, we must propagate the error (loss) back from the final loss $L$ through every single timestep.

#### **The Temporal Flow**
> **[Figure 1: Unrolled RNN & Gradient Flow]**
> ![Unrolled RNN flow](../assets/rnn_flow.png)

#### **The Chain Rule Derivation**
For a loss $L_T$ at the final timestep $T$, the gradient with respect to $W_{aa}$ is calculated as the sum of gradients at each timestep:

$$\frac{\partial L_T}{\partial W_{aa}} = \sum_{t=1}^{T} \frac{\partial L_T}{\partial a_T} \frac{\partial a_T}{\partial a_t} \frac{\partial a_t}{\partial W_{aa}}$$

---

### **3. The Mathematical Bottleneck**
The term $\frac{\partial a_T}{\partial a_t}$ is the "temporal bridge" that allows information to flow across time. It expands into a product of gradients:

$$\frac{\partial a_T}{\partial a_t} = \prod_{k=t+1}^{T} \frac{\partial a_k}{\partial a_{k-1}}$$

> **[Figure 2: The Chain Rule Path]**
> ![Chain rule derivation](../assets/bptt_derivation.png)

---

### **4. The Vanishing Gradient Problem**
The core issue of the standard RNN is how $\frac{\partial a_k}{\partial a_{k-1}}$ is calculated. Given $a_k = \tanh(W_{aa} a_{k-1} + W_{ax} x_k + b_a)$, the derivative is:

$$\frac{\partial a_k}{\partial a_{k-1}} = W_{aa}^T \cdot \text{diag}(1 - \tanh^2(z_k))$$

#### **The Role of Activation Function**
> **[Figure 3: tanh Derivative Plot]**
> ![tanh derivative plot](../assets/tanh_plot.png)

**Why the gradient vanishes:**
1. **Repeated Matrix Multiplication:** The gradient is proportional to $(W_{aa})^{T-t}$. If the largest eigenvalue of $W_{aa}$ is less than 1, the gradient decays exponentially.
2. **Saturated Activations:** The derivative of $\tanh$ is $g'(z) \in (0, 1]$. When the input $z_k$ is large (saturated), the derivative is near zero, killing the gradient signal instantly.
3. **Consequence:** $$\lim_{(T-t) \to \infty} \frac{\partial a_T}{\partial a_t} = 0$$
Information from the early steps of a sequence has zero impact on the weights, making the model "blind" to long-term context.

---

### **5. Conclusion: Moving Toward Gating**
The Vanishing Gradient problem is a physical limitation of the Simple RNN architecture. To overcome this, we move toward Gated Units (GRU/LSTM), which provide a "Linear Highway" for gradients to flow without being repeatedly diminished.