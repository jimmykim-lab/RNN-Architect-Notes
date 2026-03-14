import numpy as np
import src.molecule.rnn_forward as rnn_forward

# --- Test Script: Verifying the RNN Architecture ---

# Define RNN types  
#rnn_type = "standard"
#rnn_type = "GRU"
#rnn_type = "LSTM"
rnn_types = ["standard", "GRU", "LSTM"]

for rnn_type in rnn_types:
    # 0. initialize caches & set seed for reproducibility
    caches = []
    np.random.seed(20260313)

    # 1. Define Dimensions (The "Cockpit" Specs)
    n_x = 3      # Input size (e.g., 3 features per timestep)
    n_a = 5      # Hidden state size (Memory capacity)
    if rnn_type == "LSTM":
        n_c = 5
        n_u = 5
        n_f = 5
        n_o = 5
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
    # LSTM only - c0 shape : (n_c, m)
    if rnn_type == "LSTM":
        c0_test = np.zeros((n_c, m))
    else:
        c0_test = None

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
    elif rnn_type == "LSTM":
        parameters = {
            "Wca": np.random.randn(n_c, n_a) * 0.01,
            "Wcx": np.random.randn(n_c, n_x) * 0.01,
            "Wua": np.random.randn(n_u, n_a) * 0.01, 
            "Wux": np.random.randn(n_u, n_x) * 0.01,
            "Wfa": np.random.randn(n_f, n_a) * 0.01,
            "Wfx": np.random.randn(n_f, n_x) * 0.01,
            "Woa": np.random.randn(n_o, n_a) * 0.01,
            "Wox": np.random.randn(n_o, n_x) * 0.01,
            "Wya": np.random.randn(n_y, n_a) * 0.01,
            "bc": np.zeros((n_c, 1)),
            "bu": np.zeros((n_u, 1)),
            "bf": np.zeros((n_f, 1)),
            "bo": np.zeros((n_o, 1)),
            "by": np.zeros((n_y, 1))
        }

    # 5. Run the Engine (The Forward Pass)
    # Test with tanh for hidden and softmax for output
    y_pred, a, c, caches = rnn_forward.rnn_forward(x_test, a0_test, c0_test, parameters, act_h, act_y, rnn_type)

    ## 6. Validation (The "Stress Test" Results)
    print(f"\n--- [{rnn_type.upper()} Stress Test Results] ---")
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

    all_steps_sum = np.sum(y_pred, axis=0) # Shape: (m, T_x)
    if np.allclose(all_steps_sum, 1.0):
        print("SUCCESS: Softmax probabilities sum to 1.0 for ALL timesteps.")
    else:
        print("FAILURE: Softmax sum inconsistency detected in sequence.")

    expected_cache_len = T_x
    actual_cache_len = len(caches)

    print(f"Cache Length: {actual_cache_len} (Expected: {expected_cache_len})")

    if actual_cache_len == expected_cache_len:
        print("SUCCESS: Time-step memory (cache) is perfectly preserved for BPTT.")
    else:
        print("FAILURE: Memory leak detected! Caches were overwritten.")

    # Verify that Cell State is only present for LSTM
    if rnn_type != "LSTM" and c is not None:
        print("FAILURE: Architectural Leak! Cell state returned for non-LSTM type.")
    elif rnn_type == "LSTM" and c is None:
        print("FAILURE: Cell State missing in LSTM mode.")
    else:
        print(f"SUCCESS: Cell State management is architecturally sound for {rnn_type}.")

    # 8. Inspecting GRU/LSTM internals 
    if rnn_type == "GRU":
        # GRU cache tuple: (a_next, candidate, update_gate, reset_gate, xt, a_prev, self)
        last_cache = caches[-1]
        update_gate_val = last_cache[2] 
        
        print("[GRU Internal Inspection]")
        print(f"Update Gate shape: {update_gate_val.shape} (Expected: {n_a}, {m})")
        print(f"Update Gate Min Value: {np.min(update_gate_val):.4f} (Must be >= 0)")
        print(f"Update Gate Max Value: {np.max(update_gate_val):.4f} (Must be <= 1)")
    elif rnn_type == "LSTM":
        # LSTM cache tuple: (a_next, c_next, u_gate, f_gate, o_gate, xt, a_prev, c_prev, self)
        last_cache = caches[-1]
        u_gate_val = last_cache[2]
        f_gate_val = last_cache[3]
        o_gate_val = last_cache[4]
        c_next_val = last_cache[1]

        print("[LSTM Internal Inspection]")
        print(f"Cell State Shape (c): {c.shape} (Expected: {n_a}, {m}, {T_x})")
        print(f"Forget Gate Mean: {np.mean(f_gate_val):.4f} (Ideal: ~0.5 at init)")
        print(f"Output Gate Mean: {np.mean(o_gate_val):.4f}")
        print(f"Cell State Min/Max: {np.min(c_next_val):.4f} / {np.max(c_next_val):.4f}")

        # Stability Check: Compare the first and last cell state
        c_first = c[:, 0, 0]
        c_last = c[:, 0, -1]
        print(f"Cell State Stability (First vs Last L2 Norm): {np.linalg.norm(c_last - c_first):.4f}")