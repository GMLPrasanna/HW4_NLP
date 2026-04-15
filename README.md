# 📘 CS5760 – Natural Language Processing  
## Homework 4  

---

## 👤 Student Information  
- **Name:** Guntur Murali Lakshmi Prasanna  
- **Course:** CS5760 Natural Language Processing  
- **Semester:** Spring 2026  
- **University:** University of Central Missouri  

---

# 📌 Overview  
This assignment covers key concepts in neural networks, recurrent neural networks (RNNs), and Transformer architectures.  
It includes both theoretical analysis and practical implementations using Python.

---

# 🧠 Part I: Theory & Analytical Problems  

---

## 🔹 Q6: Multi-Input Feedforward Neural Network  

### Given:
- Inputs: x₁ = 2, x₂ = 1, x₃ = 3  
- Bias added → x = [2, 1, 3, 1]  
- One hidden layer (2 sigmoid units)  
- One sigmoid output  

---

### (a) Hidden Layer Computation  

**Pre-activations:**
- z₁ = 0.3  
- z₂ = 1.1  

**Activations (Sigmoid):**
- h₁ ≈ 0.574  
- h₂ ≈ 0.750  

---

### (b) Output Layer  

- z = 0.002  
- Final output:  
  y ≈ 0.5005  

---

### (c) Binary Cross-Entropy Loss  

- True label t = 1  
- Loss:
  
  L = -log(y) ≈ 0.692  

---

## 🔹 Q7: XOR with ReLU Network  

### Outputs:

| Input  | Output |
|--------|--------|
| (0,0)  | 0      |
| (0,1)  | 1      |
| (1,0)  | 3      |
| (1,1)  | 1 ❌   |

---

### Analysis  

- Original XOR is symmetric  
- Added hidden unit introduces **bias toward x₁**  
- Decision boundary becomes **asymmetric**  

### Conclusion  
- Model does NOT compute XOR exactly  
- Incorrect case: **(1,1)**  

---

## 🔹 Q8: Perceptron Decision Boundary  

### Given:
- w₁ = 1, w₂ = -2, b = 1  

---

### (a) Decision Boundary  

x₁ - 2x₂ + 1 = 0  

---

### (b) Classification  

| Point | Prediction | True | Result |
|------|-----------|------|--------|
| (2,1) | 1 | 1 | ✅ |
| (1,3) | 0 | 0 | ✅ |
| (3,2) | 0 | 1 | ❌ |
| (0,1) | 0 | 0 | ✅ |

- Misclassified: **(3,2)**  

---

### (c) Perceptron Loss  
- Total mistakes = **1**  

---

### (d) Weight Update  

Using learning rate η = 1:

- New weights:  
  w₁ = 4  
  w₂ = 0  

- New bias:  
  b = 2  

---

# ✏️ Short Answer Concepts  

---

## 🔹 Neural Networks  

- Non-linear activations allow learning complex relationships  
- Without them → model behaves like linear regression  

---

## 🔹 Deep vs Shallow Models  

- Deep networks learn **hierarchical features**  
- Capture more complex patterns than logistic regression  

---

## 🔹 RNN Architectures  

| Task | Type |
|------|------|
| Next-word prediction | One-to-many |
| Sentiment analysis | Many-to-one |
| NER | Many-to-many (aligned) |
| Translation | Many-to-many (unaligned) |

---

### Unrolling in RNNs  
Unrolling converts the network into repeated layers across time, enabling backpropagation through time (BPTT) with shared weights.

---

### Weight Sharing  
- Advantage: fewer parameters  
- Limitation: cannot capture position-specific behavior  

---

## 🔹 Vanishing Gradient Problem  

- Gradients shrink during backpropagation  
- Early time steps receive little learning signal  
- Model struggles with long-term dependencies  

### Solutions:
- LSTM → uses gates and cell state  
- GRU → simplified gating mechanism  

### Training Technique:
- Gradient clipping stabilizes training  

---

## 🔹 LSTM Gates  

- Forget Gate → removes irrelevant info  
- Input Gate → adds new information  
- Output Gate → controls output  

### Key Idea:
Cell state provides a **linear path for gradients**, preventing vanishing  

---

## 🔹 Self-Attention  

- Query (Q): what we search for  
- Key (K): what we match against  
- Value (V): actual information  

### Formula:

Attention(Q, K, V) = softmax(QKᵀ / √dₖ)V


### Scaling Factor:
Prevents large values → stabilizes training  

---

## 🔹 Transformer Concepts  

### Multi-Head Attention  
- Learns multiple relationships in parallel  
- Improves representation quality  

### Add & Norm  
- Residual connections → better gradient flow  
- LayerNorm → stable training  

---

## 🔹 Encoder–Decoder  

- Masked attention prevents seeing future tokens  
- Ensures proper sequence generation  

### Inference  
- Generates tokens step-by-step (autoregressive)  

---

# 💻 Part II: Programming  

---

## 🔹 Q1: Character-Level RNN  

### Objective  
Predict next character using sequence input  

### Model  
Embedding → RNN (LSTM/GRU) → Linear → Softmax  

### Training  
- Loss: Cross-Entropy  
- Optimizer: Adam  
- Teacher forcing used  

---

### Results  
- Loss curves plotted  
- Generated text with temperatures:
  - τ = 0.7 → coherent  
  - τ = 1.0 → balanced  
  - τ = 1.2 → creative/random  

---

### Observations  
- Larger hidden size → better learning  
- Longer sequences → more context  
- Higher temperature → more randomness  

---

## 🔹 Q2: Mini Transformer Encoder  

### Components  
- Embedding  
- Positional Encoding  
- Multi-head Attention  
- Feed-forward network  
- Add & Norm  

---

### Outputs  
- Contextual embeddings  
- Attention heatmaps  

---

### Insight  
Different heads capture:
- Syntax  
- Semantic relations  
- Word dependencies  

---

## 🔹 Q3: Scaled Dot-Product Attention  

### Implementation  
- Computed Q, K, V  
- Applied scaling factor  
- Used softmax  

---

### Outputs  
- Attention matrix  
- Output vectors  
- Stability comparison  

---

# ⚙️ Technologies Used  
- Python  
- PyTorch  
- NumPy  
- Matplotlib  

---

# ▶️ How to Run  

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install torch numpy matplotlib

Run each module:
python Q1_char_rnn/train.py
python Q2_transformer_encoder/transformer.py
python Q3_attention/scaled_dot_product.py
