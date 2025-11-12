import numpy as np
import sys
bs, nq, nk, dim = 2, 16, 8, 128
seq_q, seq_k = 256, 256
np.random.seed(0)
q = np.random.randn(bs, nq, seq_q, dim).astype(np.float16)
k = np.random.randn(bs, nk, seq_k, dim).astype(np.float16)
v = np.random.randn(bs, nk, seq_k, dim).astype(np.float16)
# TODO: call GGML through bindings
