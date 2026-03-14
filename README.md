# RNN-Architect-Notes

_**The Cockpit Architect: Mastering Sequential Intelligence from Scratch**_

This repository documents a rigorous journey through Sequence Models, focusing on **Architecture over Scripts**. It transforms deep-learning theory into robust, professional-grade "Cockpit" systems.

---
**Phase 1: Core Sequence Engines (March 2026)**

**1. Standard RNN & The BPTT Deep Dive (3/11, commit : 417b115, 76c0a35)**

The foundation of sequential processing, where I explored the mathematical boundaries of gradient flow.
- **BPTT (Backpropagation Through Time)**: Implemented the full derivative chain from scratch to understand how information travels (and dies) across time.
- **The Vanishing Gradient Problem**: Documented the "Why" behind the need for gated architectures. Proved mathematically why long-term dependencies fail in basic RNNs.
- **Deliverables**: Vectorized forward/backward pass for standard RNN cells.

**2. GRU (Gated Recurrent Unit) - Efficiency & Speed (3/12, commit : 9717d48)**

A streamlined approach to memory management, focusing on computational throughput without sacrificing performance.
- **Gate Logic**: Implemented Update and Reset gates to control information flow.
- **Optimization**: Designed for lower parameter counts, making it the "lightweight fighter jet" of sequence models.
- **Key Insight**: Balanced the trade-off between memory capacity and execution speed.

**3. LSTM (Long Short-Term Memory) - The Memory Highway (3/13, commit : 5322875)**
The most sophisticated engine in the 1st phase, engineered for high-fidelity long-term memory.

- * **Hardware-Aware Design:** Weights are horizontally stacked to process $a^{\langle t-1 \rangle}$ and $x^{\langle t \rangle}$ in a single unified operation, optimizing memory bandwidth.
- **The Cell State (c)**: Implemented a dedicated "Long-term Memory Highway" to persist critical information across long sequences.
- **Gating Precision**: Forget, Update, and Output gates implemented via Hadamard products for exact signal filtering.

---
**Technical Validation (The Stress Test)**
Every engine is verified through a rigorous Architect's Testbench (rnn_testbench.py):
1. **Mathematical Consistency**: Verified that Softmax output probabilities sum to exactly 1.0 for all timesteps.
2. **State Stability**: Monitored the L2 Norm of the Cell State to ensure stable signal propagation without numerical exploding/vanishing.
3. **BPTT Readiness**: Confirmed that intermediate gate values (caches) are perfectly preserved for the upcoming backward pass phase.

---
## Project Structure
```text
.
├── src/
│   ├── atom/       # Cell Architectures (Standard, GRU, LSTM)
│   ├── molecule/   # Sequence Managers (Unified rnn_forward)
│   └── shared/     # Optimized Math & Activation Functions
├── testbench/      # Stress-test & Validation Scripts
└── README.md
