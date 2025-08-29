import numpy as np

# assume Q dims are divisible by num_heads
def multi_head_attention(Q, K, V, num_heads, weights):
    """
    multi-head attention mechanism

    """
    Wq, bq = weights["Wq"].weight, weights["bq"].weight
    Wk, bk = weights["Wk"].weight, weights["bk"].weight
    Wv, bv = weights["Wv"].weight, weights["bv"].weight
    Wo, bo = weights["Wo"].weight, weights["bo"].weight

    # Q, K, V
    # x is prior state
    # no op to get here
    # Qp, Kp, Vp output
    Qp = Q @ Wq + bq
    Kp = K @ Wk + bk
    Vp = V @ Wv + bv

    Qh = split_heads(Qp, num_heads)
    Kh = split_heads(Kp, num_heads)
    Vh = split_heads(Vp, num_heads)

    qkv_split = (Qh, Kh, Vh)

    Ah, scores = attention(Qh, Kh, Vh)

    x_combined = combine_heads(Ah)

    #x_combined is prior state
    # inverse is 
    # - split()
    # - inverse attention
    #  forward is:
    #   - scores = Q @ np.transpose(K, (0, 1, 3, 2)) / dim ** 0.5
    #   - softmax(scores, axis=-1)
    #   - scores @ Vh
    #  inverse is:
    #   -
    #   -
    #   -
    # - combine heads of each individual QKV
    out = x_combined @ Wo + bo
    return out, scores, x_combined, qkv_split


def split_heads(x, num_heads):
    B, L, D = x.shape

    Dh = D // num_heads
    x = x.reshape(B, L, num_heads, Dh)

    return np.transpose(x, (0, 2, 1, 3))


def combine_heads(x):
    B, H, L, Dh = x.shape
    return np.transpose(x, (0, 2, 1, 3)).reshape(B, L, H * Dh)


def attention(Qh, Kh, Vh):
    '''
    scaled dot product attention
    '''

    dim = Qh.shape[-1]

    scores = Qh @ np.transpose(Kh, (0, 1, 3, 2)) / dim ** 0.5
    scores = softmax(scores, axis=-1)
    output = scores @ Vh

    return output, scores


def relu(x):
    '''
    ReLU activation function
    '''

    return np.maximum(0, x)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def layer_norm(x, weights):
    '''
    layer normalization
    '''
    gamma, beta = weights["gamma"].weight, weights["beta"].weight
    
    mean = np.mean(x, axis=-1, keepdims=True) 
    var = np.var(x, axis=-1, keepdims=True) 
    
    inv_std = 1.0 / np.sqrt(var + 1e-9)
    norm = (x - mean) * inv_std
    return norm * gamma + beta, norm, inv_std


def feed_forward(x, weights_ff):
    '''
    feed forward network
    '''

    ff_1 = relu(x @ weights_ff["W1"].weight + weights_ff["b1"].weight)
    ff_2 = ff_1 @ weights_ff["W2"].weight + weights_ff["b2"].weight

    return ff_2, ff_1
