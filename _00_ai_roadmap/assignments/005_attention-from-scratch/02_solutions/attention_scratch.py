"""Multi-head self-attention with causal masking, implemented in plain
Python (no NumPy/PyTorch). Verified against the checks in Assignment 005.
"""
import math
import random


def matmul(A, B):
    n, k, m = len(A), len(A[0]), len(B[0])
    return [[sum(A[i][t] * B[t][j] for t in range(k)) for j in range(m)] for i in range(n)]


def transpose(A):
    return [list(row) for row in zip(*A)]


def add_bias(X, b):
    return [[X[i][j] + b[j] for j in range(len(b))] for i in range(len(X))]


def linear(x, W, b):
    return add_bias(matmul(x, W), b)


def softmax_row(row):
    finite = [v for v in row if v != float("-inf")]
    mx = max(finite) if finite else 0
    exps = [0.0 if v == float("-inf") else math.exp(v - mx) for v in row]
    s = sum(exps)
    return [e / s for e in exps]


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = len(Q[0])
    scores = matmul(Q, transpose(K))
    scaled = [[scores[i][j] / math.sqrt(d_k) for j in range(len(scores[0]))] for i in range(len(scores))]
    if mask is not None:
        for i in range(len(scaled)):
            for j in range(len(scaled[0])):
                if mask[i][j] == 0:
                    scaled[i][j] = float("-inf")
    weights = [softmax_row(row) for row in scaled]
    return matmul(weights, V), weights


def split_heads(X, n_heads):
    d_model = len(X[0])
    d_k = d_model // n_heads
    return [[row[h*d_k:(h+1)*d_k] for row in X] for h in range(n_heads)]


def concat_heads(heads):
    seq_len = len(heads[0])
    return [sum((h[i] for h in heads), []) for i in range(seq_len)]


def multi_head_attention(x, W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o, n_heads, mask=None):
    Q, K, V = linear(x, W_q, b_q), linear(x, W_k, b_k), linear(x, W_v, b_v)
    Qh, Kh, Vh = split_heads(Q, n_heads), split_heads(K, n_heads), split_heads(V, n_heads)
    head_outputs = [scaled_dot_product_attention(Qh[h], Kh[h], Vh[h], mask=mask)[0] for h in range(n_heads)]
    return linear(concat_heads(head_outputs), W_o, b_o)


if __name__ == "__main__":
    random.seed(0)
    seq_len, d_model, n_heads = 5, 8, 2

    def rand_matrix(r, c):
        return [[random.uniform(-0.5, 0.5) for _ in range(c)] for _ in range(r)]

    def rand_vec(n):
        return [random.uniform(-0.1, 0.1) for _ in range(n)]

    x = rand_matrix(seq_len, d_model)
    W_q, b_q = rand_matrix(d_model, d_model), rand_vec(d_model)
    W_k, b_k = rand_matrix(d_model, d_model), rand_vec(d_model)
    W_v, b_v = rand_matrix(d_model, d_model), rand_vec(d_model)
    W_o, b_o = rand_matrix(d_model, d_model), rand_vec(d_model)

    causal_mask = [[1 if j <= i else 0 for j in range(seq_len)] for i in range(seq_len)]

    out_masked = multi_head_attention(x, W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o, n_heads, mask=causal_mask)

    # Q4: change position 4, confirm position 0's output is unaffected
    x2 = [row[:] for row in x]
    x2[4] = rand_vec(d_model)
    out_masked2 = multi_head_attention(x2, W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o, n_heads, mask=causal_mask)
    print("position 0 unaffected by change at position 4:",
          all(abs(a - b) < 1e-9 for a, b in zip(out_masked[0], out_masked2[0])))

    # Q5: masked vs unmasked, position by position
    out_unmasked = multi_head_attention(x, W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o, n_heads, mask=None)
    for i in range(seq_len):
        same = all(abs(a - b) < 1e-9 for a, b in zip(out_masked[i], out_unmasked[i]))
        print(f"position {i}: masked == unmasked -> {same}")
