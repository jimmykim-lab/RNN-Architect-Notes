import numpy as np
import src.molecule.rnn_forward as rnn_forward

# --- Test Script: Verifying the RNN Architecture ---

# 0. Define RNN type & initialize caches
#rnn_type = "standard"
rnn_type = "GRU"
caches = []

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
if rnn_type == "standard":
    parameters = {
        "Wax": np.random.randn(n_a, n_x) * 0.01,
        "Waa": np.random.randn(n_a, n_a) * 0.01,
        "Wya": np.random.randn(n_y, n_a) * 0.01,
        "ba": np.zeros((n_a, 1)),
        "by": np.zeros((n_y, 1))
    }
elif rnn_type == "GRU":
    parameters = {
        "Wax": np.random.randn(n_a, n_x) * 0.01,
        "Waa": np.random.randn(n_a, n_a) * 0.01,
        "Wux": np.random.randn(n_a, n_x) * 0.01, # Update gate input weight
        "Wua": np.random.randn(n_a, n_a) * 0.01, # Update gate hidden weight
        "Wrx": np.random.randn(n_a, n_x) * 0.01, # Reset gate input weight
        "Wra": np.random.randn(n_a, n_a) * 0.01, # Reset gate hidden weight
        "Wya": np.random.randn(n_y, n_a) * 0.01,
        "ba": np.zeros((n_a, 1)),
        "bu": np.zeros((n_a, 1)),
        "br": np.zeros((n_a, 1)),
        "by": np.zeros((n_y, 1))
    }

# 5. Run the Engine (The Forward Pass)
# Test with tanh for hidden and softmax for output
y_pred, a, caches = rnn_forward.rnn_forward(x_test, a0_test, parameters, act_h, act_y, rnn_type)

## 6. Validation (The "Stress Test" Results)
print(f"--- [{rnn_type.upper()} Stress Test Results] ---")
print(f"Input Sequence Shape (x): {x_test.shape}")
print(f"Hidden States Shape (a): {a.shape}")
print(f"Predictions Shape (y_pred): {y_pred.shape}")

# 7. Mathematical Consistency Check
# In a Softmax layer, the sum of probabilities across all classes must be 1.0.
prob_sum = np.sum(y_pred[:, 0, 0])
if np.isclose(prob_sum, 1.0):
    print("SUCCESS: Softmax probabilities sum to 1.0.")
else:
    print(f"FAILURE: Softmax sum is {prob_sum:.4f}")

expected_cache_len = T_x
actual_cache_len = len(caches)

print(f"Cache Length: {actual_cache_len} (Expected: {expected_cache_len})")

if actual_cache_len == expected_cache_len:
    print("SUCCESS: Time-step memory (cache) is perfectly preserved for BPTT.")
else:
    print("FAILURE: Memory leak detected! Caches were overwritten.")

# 8. Inspecting GRU internals 
if rnn_type == "GRU":
    # GRU cache tuple: (a_next, candidate, update_gate, reset_gate, xt, a_prev, self)
    last_cache = caches[-1]
    update_gate_val = last_cache[2] 
    
    print("\n[GRU Internal Inspection]")
    print(f"Update Gate shape: {update_gate_val.shape} (Expected: {n_a}, {m})")
    print(f"Update Gate Min Value: {np.min(update_gate_val):.4f} (Must be >= 0)")
    print(f"Update Gate Max Value: {np.max(update_gate_val):.4f} (Must be <= 1)")