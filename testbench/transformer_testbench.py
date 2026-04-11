import numpy as np
from src.transformer_encoder import ScaledDotProductAttention, MultiHeadAttention, AddNorm, FFN, TransformerEncoderLayer

np.random.seed(20260410)

# 1. Dimensions
T = 4          # sequence length
d_model = 8    # embedding dimension
d_k = 4        # Q, K, V dimension per head
d_ff = 32      # FFN hidden dimension (4 * d_model)
h = 2          # number of heads (d_model / d_k = 8 / 4 = 2)

# 2. Dummy input
X = np.random.randn(T, d_model)

# 3. Parameters - ScaledDotProductAttention
sdpa_parameters = {
    "WQ": np.random.randn(d_model, d_k) * 0.01,
    "WK": np.random.randn(d_model, d_k) * 0.01,
    "WV": np.random.randn(d_model, d_k) * 0.01,
}

# 4. Parameters - MultiHeadAttention
mha_parameters = {
    **{f"head_{i}": {
        "WQ": np.random.randn(d_model, d_k) * 0.01,
        "WK": np.random.randn(d_model, d_k) * 0.01,
        "WV": np.random.randn(d_model, d_k) * 0.01,
    } for i in range(h)},
    "WO": np.random.randn(h * d_k, d_model) * 0.01,
}

# 5. Parameters - AddNorm
addnorm_parameters = {
    "gamma": np.ones(d_model),
    "beta": np.zeros(d_model),
}

# 6. Parameters - FFN
ffn_parameters = {
    "W1": np.random.randn(d_model, d_ff) * 0.01,
    "W2": np.random.randn(d_ff, d_model) * 0.01,
    "b1": np.zeros(d_ff),
    "b2": np.zeros(d_model),
}

# 7. Parameters - TransformerEncoderLayer
encoder_parameters = {
    "mha": mha_parameters,
    "addnorm_1": addnorm_parameters,
    "ffn": ffn_parameters,
    "addnorm_2": addnorm_parameters,
}

# 8. Run - ScaledDotProductAttention
sdpa = ScaledDotProductAttention(sdpa_parameters)
sdpa_output = sdpa.forward(X)

# 9. Run - MultiHeadAttention
mha = MultiHeadAttention(mha_parameters, h)
mha_output = mha.forward(X)

# 10. Run - AddNorm (after MHA)
addnorm_1 = AddNorm(addnorm_parameters)
addnorm_1_output = addnorm_1.forward(X, mha_output)

# 11. Run - FFN
ffn = FFN(ffn_parameters)
ffn_output = ffn.forward(addnorm_1_output)

# 12. Run - AddNorm (after FFN)
addnorm_2 = AddNorm(addnorm_parameters)
addnorm_2_output = addnorm_2.forward(addnorm_1_output, ffn_output)

# 13. Run - TransformerEncoderLayer
encoder = TransformerEncoderLayer(encoder_parameters, h)
encoder_output = encoder.forward(X)

# 14. Validation
print("--- [ScaledDotProductAttention] ---")
print(f"Input shape: {X.shape}")
print(f"Output shape: {sdpa_output.shape}")             # Expected: (T, d_k)

print("\n--- [MultiHeadAttention] ---")
print(f"Input shape: {X.shape}")
print(f"Output shape: {mha_output.shape}")              # Expected: (T, d_model)

print("\n--- [AddNorm 1 - after MHA] ---")
print(f"Input shape: {X.shape}")
print(f"Output shape: {addnorm_1_output.shape}")        # Expected: (T, d_model)

print("\n--- [FFN] ---")
print(f"Input shape: {addnorm_1_output.shape}")
print(f"Output shape: {ffn_output.shape}")              # Expected: (T, d_model)

print("\n--- [AddNorm 2 - after FFN] ---")
print(f"Input shape: {addnorm_1_output.shape}")
print(f"Output shape: {addnorm_2_output.shape}")        # Expected: (T, d_model)

print("\n--- [TransformerEncoderLayer] ---")
print(f"Input shape: {X.shape}")
print(f"Output shape: {encoder_output.shape}")          # Expected: (T, d_model)

# LayerNorm validation: per-token mean≈0, var≈1
mean_check = np.mean(encoder_output, axis=-1)
var_check = np.var(encoder_output, axis=-1)
print(f"\nPer-token mean (should be ≈0): {mean_check}")
print(f"Per-token var (should be ≈1): {var_check}")

# Shape consistency validation: Encoder input/output shape must be identical
assert encoder_output.shape == X.shape, "FAILURE: Encoder output shape mismatch"
print("\nSUCCESS: Encoder input/output shape consistent.")