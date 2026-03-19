# [Architecture Review] Attention Is All You Need: The End of Sequential DNA - [Link to Paper : Attention IS All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

- **Date**: 2026-03-16 (Start)
- **Status**: 
    - [2026-03-16] Phase 1 Complete: (Paper Section 1 & 2 analyzed; Markdown Section I & III documented) 
    - [2026-03-18 Target]: Section II (Transformer Core Architecture) & Self-Attention Mathematics.
- **Goal**: Deconstruct the Transformer architecture to achieve the **Theoretical Optimum**.

---

## I. The Motivation: Why RNNs Must Die
### **The Sequential Bottleneck**:
- *"**Sequential nature precludes parallelization** within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples."* 
- **Architect's View** : The core limitation of RNN architecture is their sequential nature over time. The GRU and LSTM have been established as state-of-the-art approaches but the fundamental constraint of sequential computation remains. The sequential chain ($a^{\langle t \rangle} = f(a^{\langle t-1 \rangle}, x^{\langle t \rangle})$) creates an optimization bottleneck (Vanishing Gradient) and a computational bottleneck ($O(n)$ serial execution). Other approaches like Bidirectional RNN and Deep RNN require 2x computation costs and 100% latency or double vanishing issue which make it unsuitable for real-time monitoring and inefficient for scaling. Modern GPUs are built for SIMD (Single Instruction, Multiple Data) but this limitation forces serial execution so that low utilization is inevitable. And if sequence is getting longer and longer, the bottleneck of memory bandwidth is getting critical since it should send/receive a bunch of hidden state due to $O(n)$ dependency. It definitely degrades latency and performance of system.
### **The Distance Problem**: 
- *"The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU, **ByteNet** and **ConvS2S**, all of which use **convolutional neural networks** as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions **grows in the distance between positions**, **linearly for ConvS2S and logarithmically for ByteNet**. This makes it more difficult to learn dependencies between distant positions."*
### **The $O(1)$ Revolution**: 
- *"In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2."*
- **Architect's View** : This paper states **Transformer** can reduce the number of operations to a **constant number** unlike legacy models where complexity grows with distance between positions. It can solve the core limitation of RNN architecture which is sequential computation so that optimization and computation bottleneck can be resolved correspondingly. And it would be helpful for the bottleneck of memory bandwidth dramatically. In Table 1 in paper, this innovation is quantized as **Maximum Path Length ($O(1)$)**. It means all information is connected by single-step regardless of length of sequence. This is core theory to destroy dependency of RNN on $O(n)$. 

---

## II. The Engine: Transformer Core Architecture

### **The Input Pipeline: The Materials (Material Preparation)**
> **Focus**: How to transform raw data into "Intelligent Material" before it enters the Attention engine.

#### **Embeddings (Creating Semantic Coordinates)**:
- *"In our model, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension $d_{model}$."*
- *"In the embedding layers, we multiply those weights by $\sqrt{d_{model}}$.​"*
- **Mechanism**: Learned embeddings to convert input tokens to vectors of dimension $d_{model} = 512$.
- **Architect's View**: In this system, we use shared weights between input/output embeddings to optimize memory bandwidth. A crucial detail is the **scaling factor $\sqrt{d_{model}}$**, which prevents the dot-product attention from growing too large in magnitude, ensuring stable gradient flow. This is the first "knob" to ensure **Engineering Determinism**.
Architect's View: 
    From Tokens to Tensors: 임베딩은 단순한 치환이 아니라, 이산적인 데이터를 연속적인 벡터 공간(Vector Space)으로 맵핑하는 과정입니다. 이는 지미님이 설계할 RAG 시스템에서 **'지식의 좌표'**를 생성하는 기초가 됩니다.
    The  d model Scaling: 왜 굳이 제곱근 값을 곱할까요? 이는 차원이 커질수록 Dot-product 값이 커져 Softmax의 기울기가 소실되는 것을 방지하기 위한 수학적 안전장치입니다. 아키텍트에게는 시스템의 **Numerical Stability(수치적 안정성)**를 확보하는 '결정론적 설계'의 일환으로 읽힙니다.

#### **Positional Encoding (Time without a Clock)**:
- *"Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence."*
- *"We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than those encountered during training."*
- **Mechanism**: Injecting relative/absolute position using Sine and Cosine functions of different frequencies.
- **Why Sinusoids?**: Unlike RNNs that learn position through recurrence ($O(n)$), Sinusoids allow the model to *calculate* any position in $O(1)$. It allows the model to extrapolate to sequence lengths longer than those encountered during training.
- **Architect's View**: This is "Time without a Clock." We provide the spatial coordinates to the model so it can perform SIMD operations across the entire sequence while still understanding the "Sequential DNA" of the hardware logic or waveform.
Architect's View:
    The End of Recurrence: RNN은 '시간(Order)'을 처리하기 위해 $O(n)$의 순차 연산을 강제했습니다. 하지만 Transformer는 위치 정보를 **함수값(Sinusoid)**으로 계산하여 입력값에 '더해버림'으로써, 순서를 지키면서도 연산은 $O(1)$로 병렬화하는 혁신을 이뤘습니다.
    Extrapolation Power: 학습하지 않은 긴 시퀀스(Longer Sequence)에 대해서도 유연하게 대처할 수 있는 '주기 함수'를 선택했다는 점은, 시스템의 **Scalability(확장성)**를 설계 단계에서 이미 고려했음을 보여줍니다. 칩 설계 데이터처럼 시퀀스가 매우 긴 도메인에서 이 방식은 필수적입니다.

### **The Attention Mechanism: Multi-Head Attention**
*(이 부분은 논문 3.2절을 읽고 이어서 작성하시면 됩니다.)*

---

## III. Efficiency Quantization (Complexity Analysis)
### **Table 1: Comparison of Layer Types**
| Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length |
| :--- | :--- | :--- | :--- |
| **Self-Attention** | $O(n^2 \cdot d)$ | $O(1)$ | **$O(1)$** |
| **Recurrent** | $O(n \cdot d^2)$ | $O(n)$ | **$O(n)$** |
| **Convolutional** | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k(n))$ |
| **Self-Attention (restricted)** | $O(r \cdot n \cdot d)$ | $O(1)$ | $O(n/r)$ |

- **Architect's View**: Even with the quadratic complexity $O(n^2 \cdot d)$, the reduction of **Sequential Operations and Path Length to $O(1)$** is the masterstroke for hardware parallelization. It transforms the sequential "bottleneck" into a parallel "superhighway."

### **Technical Deep-Dive**
- **[Complexity per Layer] $n$ vs $d$:**: In Transformer, we trade sequence length complexity ($n^2$) for hidden state complexity ($d^2$). Since the dimensionality $d$ is often significantly larger than the sequence length $n$ in most practical scenarios, **this is a highly efficient trade for overall computational throughput**.
- **[Sequential Operations] SIMD Utilization:** The $O(1)$ sequential operations allow modern GPUs to leverage their full **SIMD (Single Instruction, Multiple Data)** potential. Unlike RNNs that force serial execution and leave GPU cores idle, Transformer ensures that computational resources are fully saturated by dispatching operations in parallel.
- **[Maximum Path Length] Information Teleportation:** The $O(1)$ path length provides "Information Teleportation," allowing gradients to flow directly between any two positions regardless of their distance. This solves the **Vanishing Gradient** issue at the architectural level, ensuring high-fidelity signal propagation throughout the network.

---

## IV. Systematic Evaluation (DSO.ai Strategic Insight)
*SOP의 핵심: 이 엔진을 실제 우리 제품에 어떻게 이식할 것인가.*
- **Reliability Assessment**: Training stability vs. RNN's vanishing gradient.
- **Proposed Application**: How Attention can optimize Chip Design Layout bottlenecks.

---

## V. Final Verdict
- **Summary**: (예: Transformer is not just a model; it's a hardware-aware optimization paradigm.)

---
## References & Evidence
- [Link to Notes : BPTT and Vanishing Gradient](01_Basic%20RNN_BPTT_and_Vanishing_Gradient.md) 
- [Link to Notes : Bidirectional & Deep RNN](02_RNN_Final_Bridge_Analysis.md)