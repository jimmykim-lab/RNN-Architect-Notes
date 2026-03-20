# [Architecture Review] Attention Is All You Need: The End of Sequential DNA - [Link to Paper : Attention IS All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

- **Date**: 2026-03-16 (Start) - 2026-03-19 (Ongoing)
- **Status**: 
    - ✅ [2026-03-16] **Section I & III Complete**: Analysis of Section 1 Introduction, Section 2 Background, and Complexity ($O(n)$ vs $O(1)$ via Table 1).
    - ✅ [2026-03-18] **Section II (Part 1) Complete**: The Input Pipeline (Section 3.4 Embeddings & 3.5 Positional Encoding).
    - 🎯 [2026-03-20 Target]: **Section II (Part 2)**: Core Architecture & Attention Mechanism (Section 3.1 Model Architecture & 3.2 Multi-Head Attention).
- **Goal**: Deconstruct the Transformer architecture to achieve the **Theoretical Optimum**.

---

## I. The Motivation: Why RNNs Must Die (Section 1 Introduction, Section 2 Background)

### **The Sequential Bottleneck** (Section 1 Introduction):
- *"**Sequential nature precludes parallelization** within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples."* 

- **Architect's View** : The core limitation of RNN architecture is their sequential nature over time. The GRU and LSTM have been established as state-of-the-art approaches but the fundamental constraint of sequential computation remains. The sequential chain ($a^{\langle t \rangle} = f(a^{\langle t-1 \rangle}, x^{\langle t \rangle})$) creates an optimization bottleneck (Vanishing Gradient) and a computational bottleneck ($O(n)$ serial execution). Other approaches like Bidirectional RNN and Deep RNN require 2x computation costs and 100% latency or double vanishing issue which make it unsuitable for real-time monitoring and inefficient for scaling. Modern GPUs are built for SIMD (Single Instruction, Multiple Data) but this limitation forces serial execution so that low utilization is inevitable. And if sequence is getting longer and longer, the bottleneck of memory bandwidth is getting critical since it should send/receive a bunch of hidden state due to $O(n)$ dependency. It definitely degrades latency and performance of system.

### **The Distance Problem** (Section 2 Background): 
- *"The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU, **ByteNet** and **ConvS2S**, all of which use **convolutional neural networks** as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions **grows in the distance between positions**, **linearly for ConvS2S and logarithmically for ByteNet**. This makes it more difficult to learn dependencies between distant positions."*

### **The $O(1)$ Revolution** (Section 2 Background): 
- *"In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2."*

- **Architect's View** : This paper states **Transformer** can reduce the number of operations to a **constant number** unlike legacy models where complexity grows with distance between positions. It can solve the core limitation of RNN architecture which is sequential computation so that optimization and computation bottleneck can be resolved correspondingly. And it would be helpful for the bottleneck of memory bandwidth dramatically. In Table 1 in paper, this innovation is quantized as **Maximum Path Length ($O(1)$)**. It means all information is connected by single-step regardless of length of sequence. This is core theory to destroy dependency of RNN on $O(n)$. 

---

## II. The Engine: Transformer Core Architecture (Section 3 Model Architecture)

### **The Input Pipeline: The Materials (Material Preparation)** (Section 3.4 Embedding and Softmax, Section 3.5 Positional Encoding)

#### **Embeddings (Creating Semantic Coordinates)** (Section 3.4 Embedding and Softmax):
- *"In our model, we use **learned embeddings** to convert the input tokens and output tokens to **vectors of dimension $d_{model}$**."*
- *"In the embedding layers, we multiply those **weights by $\sqrt{d_{model}}$**.​"*

- **Mechanism**: 
    - **Embedding**: Learned embeddings is used to convert input tokens and output tokens to **vectors of dimension $d_{model}$** to have Semantic Density.
    - **Transformation & Softmax**: The embedded vector is multiplied by a massive weight matrix W, transforming it onto a high-dimensional space matching the full vocabulary size. The Softmax normalizes all scores into a range between 0 and 1, ensuring the total sum equals exactly 1.0.
    With Transformation and Softmax, the system can get predicted next-token probabilities.
    - **Weights Sharing**: The Same Weight Matrix is shared between input and output embedding layers for efficiency **since only difference of these two layers is direction** so should share context map.
    - **Scaling Weights**: The Weight Matrix for input and output embedding layers is multiplied by **$\sqrt{d_{model}}$**.

- **Architect's View**: 
    - **From Tokens to Tensors**: The **Embeddings** not merely a simple substitution, but a process of mapping discrete data into a **Continuous Vector Space** containing Semantic Density for relation between tokens. This serves as the foundation for creating the **Coordinates of Knowledge** in the RAG system.
    - **The Bridge to Human Language**: The **Transformation & Softmax** is a **Digital-to-Analog** converter that maps the abstract vector space back into a discrete word space. It is a process of asking, **'How closely does this abstract meaning ($d_{model}$) resemble each word in our vocabulary ($V$)?'** This is the moment where 'Logits' are assigned to each word and the final stage of the Engineering Determinism. It converts abstract signals into a probabilistic certainty.
    - **Efficiency Logic**: The **Sharing Weights** can maximize **memory efficiency** by reducing number of parameters. This eliminates the need for redundant storage so slashes memory bandwidth in half. From the perspective of the hardware accelerator, this is a masterstroke that drastically reduces VRAM consumption.
    - **Numerical Governance**: Multiplying the embedding weights by $\sqrt{d_{model}}$ is not mere arithmetic; it is a form of Numerical Governance for **Engineering Determinism** designed to **prevent loss of context** and **maintain the discriminative power of Softmax** in high-dimensional spaces.
        - **Signal-to-Noise Ratio(SNR) Control**: In the Transformer architecture, embeddings ($E$) and positional information ($PE$) are combined through summation ($E + PE$). Learned embeddings often maintain very small magnitudes depending on their initialization. If the embedding values are too minute, the original 'semantic meaning' of the word is drowned out by the 'noise' of the positional information the moment they are merged. By scaling the embeddings by $\sqrt{d_{model}}$, it adequately amplify the vector magnitude. This ensures the word's inherent meaning remains distinct and robust even after being integrated with positional data.
        - **Gradient Preservation**: As the vector dimension ($d_{model}$) increases, the dot-product values of embedding vectors tend to grow exponentially and the variance of the dot-product between two vectors grows in proportion to $d_{model}$. Excessive dot-product magnitudes cause the Softmax function to saturate, driving outputs to extreme values of 0 or 1. This leads to Gradient Vanishing, where the derivative approaches zero, effectively halting the learning process. Dividing by or pre-multiplying by $\sqrt{d_{model}}$ acts as a 'mathematical damper' that normalizes this variance to 1, ensuring the Softmax input remains within the 'Active Region' for effective learning.
        Pre-scaling the embeddings in Section 3.4 is a preemptive measure to control potential 'numerical explosion' during the Attention operations (Section 3.2.1).

#### **Positional Encoding (Injecting Temporal DNA)** (Section 3.5 Positional Encoding):
- *"Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must **inject some information about the relative or absolute position** of the tokens in the sequence."*
- *"We chose the **sinusoidal version** because it may allow the model to **extrapolate** to sequence lengths longer than those encountered during training."*

- **Mechanism - The "Mathematical Clock" System**:
    - **Mathematical Timestamp**: Because Transformers process all tokens simultaneously (Parallelism), they lack a natural sense of "time." Positional Encoding acts as a **"Digital Seat Ticket,"** assigning a unique mathematical coordinate to every token so the model knows "who sits where."
    - **The Sinusoidal Formula**: Each word is equipped with 512 different "clocks" (dimensions). The clocks at the beginning spin fast (high frequency), while those at the end spin slowly (low frequency). This creates a **Unique Fingerprint** for every position:
        $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
        $$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
    - **Linear Relationship**: Using Sine and Cosine allows the model to calculate the distance between words easily. For any fixed offset $k$, $PE_{pos+k}$ can be represented as a **linear function** of $PE_{pos}$. It’s like knowing that "15 minutes later" is always a $90^\circ$ turn on the clock face, no matter what time it is now.
    - **Extrapolation Capability**: Unlike learned positions, this functional approach allows the model to **"calculate"** the position of the 1,001st word even if it was only trained on 500 words. It follows the continuous wave pattern.

- **Architect's View**: 
    - **Time without a Clock**: In RNNs, time was a physical bottleneck ($O(n)$). In Transformers, time is treated as a **Spatial Coordinate**. We provide a "Compass of Time" so the system can perform SIMD (Single Instruction, Multiple Data) operations across the entire sequence while still respecting the **"Sequential DNA"** of the data.
    - **The End of Recurrence ($O(1)$ Revolution)**: By *calculating* position instead of *waiting* for it, we transform temporal dependency into a constant-time mathematical addition. This is the ultimate **"Architectural Bypass"** for hardware scaling on GPUs.
    - **Information Fusion ($E + PE$)**: We "layer" these positional waves on top of the semantic embeddings. This is a high-efficiency technique that keeps the data compact for memory bandwidth. 
    - **The Necessity of Scaling**: To ensure the "Time Stamp" ($PE$) doesn't drown out the "Word Meaning" ($E$), we rely on the **Numerical Governance** established in Section 3.4 (scaling by $\sqrt{d_{model}}$). This keeps the semantic signal robust even after the temporal DNA is injected.


### **The Processing Core: System Assembly & Intelligence** (Section 3.1 Encoder and Decoder Stacks, Section 3.2 Attention) 
*🎯 [2026-03-20 Target]: **Section II (Part 2)**: Core Architecture & Attention Mechanism (Section 3.1 Model Architecture & 3.2 Multi-Head Attention)*
#### **Model Architecture (The Framework: The Multi-Stage Stack)** (Section 3.1 Encoder and Decoder Stacks)
Residual Connection: "The Highway System" - 신호 감쇄를 막는 고속도로.
LayerNorm: "The Voltage Regulator" - 각 층의 출력을 표준 범위 내로 가두는 전압 조절기.
#### **The Attention Mechanism: Multi-Head Attention** (Section 3.2 Attention)
Multi-Head: "Parallel Sensory Filters" - 같은 데이터를 8개의 서로 다른 시각(헤드)으로 동시에 필터링하는 병렬 센서.
Scaled Dot-Product: 3.4절에서 언급한 "Variance Damper" 로직이 여기서 실제 연산으로 구현됨을 강조.

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
- [Link to Notes : Sematic Embedding](04_Semantic_Embedding_and_RAG_Logic.md)
- [Link to Article : Understanding Positional Encoding in Transformers](https://erdem.pl/2021/05/understanding-positional-encoding-in-transformers)
- [Link to Blog : Jay Alammar - The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Link to YouTube : StatQuest - Transformer Neural Networks, ChatGPT's Foundation](https://youtu.be/zxQyTK8quyY?si=UxMW5GcgRg50IUKk)
