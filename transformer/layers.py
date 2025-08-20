import numpy as np
import scipy

# assume Q dims are divisible by num_heads

def multi_head_attention(Q, K, V, num_heads):   
    '''
    multi-head attention mechanism

    would like to compute each head in parallel
    '''


    attention_heads =[]
    heads_dim = Q.shape[-1] // num_heads

    for i in range(0, Q.shape[-1], heads_dim):
        attention_heads.append( attention(Q[:, i : i + heads_dim], K[:, i : i + heads_dim], V[:, i : i + heads_dim]) )

    # flatten the attention heads
    multihead_attention = np.concatenate(attention_heads, axis=-1)

    return multihead_attention


def attention(Q, K, V):
    '''
    scaled dot product attention
    '''

    dim = Q.shape[-1]

    scores = Q @ K.T / dim ** 0.5
    scores = scipy.special.softmax(scores, axis=-1)
    output = scores @ V

    return output


def relu(x):
    '''
    ReLU activation function
    '''

    return np.maximum(0, x)


def layer_norm(x):
    '''
    Layer normalization
    '''

    mean = np.mean(x, axis = -1, keepdims=True)
    std = np.std(x, axis = -1, keepdims=True)
    epsilon = 1e-9

    return (x - mean) / (std + epsilon)

def feed_forward(x):
    pass
    '''
    Feed forward network
    '''


def transformer_block(Q, K, V, num_heads):
    '''
    Single transformer block
    '''

    x = multi_head_attention(Q, K, V, num_heads)
    x = Q + x
    x = layer_norm(x)

    ff = feed_forward(x)
    ff = ff + x
    ff = layer_norm(ff)

    return ff