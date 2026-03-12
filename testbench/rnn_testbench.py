import numpy as np
import src.molecule.rnn_forward as rnn_forward

# --- Test Script: Verifying the RNN Architecture ---

# 1. Define Dimensions (The "Cockpit" Specs)
n_x = 3      # Input size (e.g., 3 features per timestep)
n_a = 5      # Hidden state size (Memory capacity)
n_y = 2      # Output size (e.g., 2 classes for classification)
m = 1        # Batch size (Single sequence for now)
T_x = 4      # Number of timesteps (Sequence length)

# 2. Define Activation Function
# act_h: string, hidden state activation - "tanh", "relu", "sigmoid"
# act_y: string, output layer activation - "sigmoid", "softmax"
act_h = "tanh"
act_y = "softmax"

# 3. Initialize Dummy Data
# x shape: (n_x, m, T_x)
x_test = np.random.randn(n_x, m, T_x)
# a0 shape: (n_a, m)
a0_test = np.zeros((n_a, m))

# 4. Initialize Parameters (Weights & Biases)
parameters = {
    "Wax": np.random.randn(n_a, n_x) * 0.01,
    "Waa": np.random.randn(n_a, n_a) * 0.01,
    "Wya": np.random.randn(n_y, n_a) * 0.01,
    "ba": np.zeros((n_a, 1)),
    "by": np.zeros((n_y, 1))
}

# 5. Run the Engine (The Forward Pass)
# Test with tanh for hidden and softmax for output
y_pred, a = rnn_forward.rnn_forward(x_test, a0_test, parameters, act_h, act_y)

# 6. Validation (The "Stress Test" Results)
print("--- [RNN Stress Test Results] ---")
print(f"Input Sequence Shape (x): {x_test.shape} (Expected: {n_x}, {m}, {T_x})")
print(f"Hidden States Shape (a): {a.shape} (Expected: {n_a}, {m}, {T_x})")
print(f"Predictions Shape (y_pred): {y_pred.shape} (Expected: {n_y}, {m}, {T_x})")

# 7. Mathematical Consistency Check
# In a Softmax layer, the sum of probabilities across all classes must be 1.0.
t_idx = 0
prob_sum = np.sum(y_pred[:, 0, t_idx])
print(f"Softmax Sum Check (t={t_idx}): {prob_sum:.4f} (Expected: 1.0000)")

if np.isclose(prob_sum, 1.0):
    print("SUCCESS: The RNN engine is mathematically sound.")
else:
    print("FAILURE: There is a leak in your probability logic.")