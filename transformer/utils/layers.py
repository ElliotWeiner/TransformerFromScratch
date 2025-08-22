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

    Qp = Q @ Wq + bq 
    Kp = K @ Wk + bk
    Vp = V @ Wv + bv

    Qh = split_heads(Qp, num_heads)
    Kh = split_heads(Kp, num_heads)
    Vh = split_heads(Vp, num_heads)

    Ah = attention(Qh, Kh, Vh)

    M = combine_heads(Ah)
    out = M @ Wo + bo
    return out


def split_heads(x, num_heads):
    B, L, D = x.shape

    Dh = D // num_heads
    x = x.reshape(B, L, num_heads, Dh)

    return np.transpose(x, (0, 2, 1, 3))


def combine_heads(x):
    B, H, L, Dh = x.shape
    return np.transpose(x, (0, 2, 1, 3)).reshape(B, L, H * Dh)


def attention(Q, K, V):
    '''
    scaled dot product attention
    '''

    dim = Q.shape[-1]

    scores = Q @ np.transpose(K, (0, 1, 3, 2)) / dim ** 0.5
    scores = softmax(scores, axis=-1)
    output = scores @ V

    return output


def relu(x):
    '''
    ReLU activation function
    '''

    return np.maximum(0, x)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


# TODO: this should be learnable
def layer_norm(x):
    '''
    layer normalization
    '''

    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + 1e-9)


def feed_forward(x, weights):
    '''
    feed forward network
    '''

    ff_1 = relu(x @ weights["W1"].weight + weights["b1"].weight)
    ff_2 = ff_1 @ weights["W2"].weight + weights["b2"].weight

    return ff_2


def transformer_block(Q, K, V, num_heads, weights_attention, weights_linear):
    '''
    single transformer block
    '''

    x = multi_head_attention(Q, K, V, num_heads, weights_attention)
    x = Q + x
    x = layer_norm(x)

    ff = feed_forward(x, weights_linear)
    ff = ff + x
    ff = layer_norm(ff)

    return ff