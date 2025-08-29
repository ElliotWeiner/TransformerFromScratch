from .layers import *
from .utils import random_init

from dataclasses import dataclass
import numpy as np
        
@dataclass
class param:
    weight: np.ndarray
    grad_weight: np.ndarray
    m: np.ndarray = None  # first moment
    v: np.ndarray = None  # second moment


class ViT_embedding:
    '''
    vision transformer embedding layer
    '''
    def __init__(self, input_shape, embed_dim):
        
        self.input_shape = input_shape
        self.embed_dim = embed_dim
        self.patch_size = 16
        patch_dim = self.patch_size * self.patch_size * input_shape[-1]

        self.weights = {
            "W": param(random_init((patch_dim, embed_dim)), np.zeros((patch_dim, embed_dim)), np.zeros((patch_dim, embed_dim)), np.zeros((patch_dim, embed_dim))),
            "b": param(random_init((embed_dim,)), np.zeros((embed_dim,)), np.zeros((embed_dim,)), np.zeros((embed_dim,)))
        }
        
        self.H, self.W, self.D = self.input_shape
        self.num_patches_h = self.H // self.patch_size
        self.num_patches_w = self.W // self.patch_size
        self.L = self.num_patches_h * self.num_patches_w  # total patches

        self.input_patch_size = self.patch_size * self.patch_size * self.D

        # add CLS token
        self.cls_token = {
            "W": param(random_init((1, 1, embed_dim)), np.zeros((1, 1, embed_dim)), np.zeros((1, 1, embed_dim)), np.zeros((1, 1, embed_dim)))
        }

        # add position embedding
        self.pos_embedding = {
            "W": param(random_init((1, self.L + 1, embed_dim)), np.zeros((1, self.L + 1, embed_dim)), np.zeros((1, self.L + 1, embed_dim)), np.zeros((1, self.L + 1, embed_dim)))
        }

        self.weight_names = ['weights', 'cls_token', 'pos_embedding']

    def forward(self, x):
        '''
        forward pass
        '''
        self.x = x

        B = x.shape[0]

        # patchify to 16x16
        x = x.reshape(B, self.num_patches_h, self.patch_size, self.num_patches_w, self.patch_size, self.D)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        # flatten patches to be (B, L, patch_size * patch_size * D)
        x = x.reshape(B, self.num_patches_h * self.num_patches_w, self.input_patch_size)

        W, b = self.weights["W"].weight, self.weights["b"].weight

        out = x @ W + b

        out = np.concatenate([np.tile(self.cls_token["W"].weight, (B, 1, 1)), out], axis=1)
        out += self.pos_embedding["W"].weight
        

        return out

    def backward(self, grad_output):
        '''
        backward pass
        '''
        self.pos_embedding["W"].grad_weight += np.sum(grad_output, axis=0, keepdims=True)

        self.cls_token["W"].grad_weight += np.sum(grad_output[:,0,:], axis=0, keepdims=True)


        self.weights["b"].grad_weight += np.sum(grad_output[:,1:,:], axis=(0,1))

        
        self.weights["W"].grad_weight += self.x.reshape(-1, self.input_patch_size).T  @ grad_output[:,1:,:].reshape(-1, self.embed_dim)

        # deepest, no need to return anything
        return 


class Transformer:
    '''
    transformer model
    '''

    def __init__(self, num_heads, input_shape):
        self.num_heads = num_heads

        self.Dh = input_shape[-1] // num_heads

        self.weights_attention = {
            "Wq": param(random_init((input_shape[-1], input_shape[-1])), np.zeros((input_shape[-1], input_shape[-1])), np.zeros((input_shape[-1], input_shape[-1])), np.zeros((input_shape[-1], input_shape[-1]))),
            "bq": param(random_init((input_shape[-1],)), np.zeros((input_shape[-1],)), np.zeros((input_shape[-1],)), np.zeros((input_shape[-1],))),
            
            "Wk": param(random_init((input_shape[-1], input_shape[-1])), np.zeros((input_shape[-1], input_shape[-1])), np.zeros((input_shape[-1], input_shape[-1])), np.zeros((input_shape[-1], input_shape[-1]))),
            "bk": param(random_init((input_shape[-1],)), np.zeros((input_shape[-1],)), np.zeros((input_shape[-1],)), np.zeros((input_shape[-1],))),

            "Wv": param(random_init((input_shape[-1], input_shape[-1])), np.zeros((input_shape[-1], input_shape[-1])), np.zeros((input_shape[-1], input_shape[-1])), np.zeros((input_shape[-1], input_shape[-1]))),
            "bv": param(random_init((input_shape[-1],)), np.zeros((input_shape[-1],)), np.zeros((input_shape[-1],)), np.zeros((input_shape[-1],))),

            "Wo": param(random_init((input_shape[-1], input_shape[-1])), np.zeros((input_shape[-1], input_shape[-1])), np.zeros((input_shape[-1], input_shape[-1])), np.zeros((input_shape[-1], input_shape[-1]))),
            "bo": param(random_init((input_shape[-1],)), np.zeros((input_shape[-1],)), np.zeros((input_shape[-1],)), np.zeros((input_shape[-1],))),
        }

        self.d_model = input_shape[-1]
        self.d_ff = 4 * self.d_model

        self.weights_linear = {
            "W1": param(random_init((self.d_model, self.d_ff)), np.zeros((self.d_model, self.d_ff)), np.zeros((self.d_model, self.d_ff)), np.zeros((self.d_model, self.d_ff))),
            "b1": param(random_init((self.d_ff,)), np.zeros((self.d_ff,)), np.zeros((self.d_ff,)), np.zeros((self.d_ff,))),
            "W2": param(random_init((self.d_ff, self.d_model)), np.zeros((self.d_ff, self.d_model)), np.zeros((self.d_ff, self.d_model)), np.zeros((self.d_ff, self.d_model))),
            "b2": param(random_init((self.d_model,)), np.zeros((self.d_model,)), np.zeros((self.d_model,)), np.zeros((self.d_model,)))
        }

        self.weights_norm_att = {
            "gamma": param(np.ones((self.d_model,)), np.zeros((self.d_model,)), np.zeros((self.d_model,)), np.zeros((self.d_model,))),
            "beta": param(np.zeros((self.d_model,)), np.zeros((self.d_model,)), np.zeros((self.d_model,)), np.zeros((self.d_model,))),
        }

        self.weights_norm_ff = {
            "gamma": param(np.ones((self.d_model,)), np.zeros((self.d_model,)), np.zeros((self.d_model,)), np.zeros((self.d_model,))),
            "beta": param(np.zeros((self.d_model,)), np.zeros((self.d_model,)), np.zeros((self.d_model,)), np.zeros((self.d_model,))),
        }

        self.weight_names = ['weights_attention', 'weights_linear', 'weights_norm_att', 'weights_norm_ff']

    def forward(self, Q, K, V):
        '''
        forward pass

        for each learned parameter, i need:
        - the reverse operation of what brought has been done since the last learned parameter
        - the x_in to each x @ w + b
        - the output
        '''
        self.qkv = (Q, K, V)
        
        # Q before
        # inv_std1 is part of prior operation
        # prescale1 is x_in
        x_norm, prescale, inv_std = layer_norm(Q, self.weights_norm_att)
        self.prescale1 = prescale
        self.inv_std1 = inv_std
        self.normed_1 = x_norm

        # Q, K, V
        # x_norm is prior state
        # no op to get here
        # Qp, Kp, Vp output

        # x_combined is prior state
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
        x, attention_scores, x_combined, qkv_split = multi_head_attention(x_norm, x_norm, x_norm, self.num_heads, self.weights_attention)
        self.softmax_scores = attention_scores
        self.qkv_split = qkv_split
        self.x_combined = x_combined
        x = Q + x

        # x before
        # inv_std2 is part of prior operation
        # prescale2 is x_in
        x_norm, prescale, inv_std = layer_norm(x, self.weights_norm_ff)
        self.prescale2 = prescale
        self.inv_std2 = inv_std
        self.normed_2 = x_norm

        # x_norm before and is x_in
        # no prior operation
        # intermediate_x after and x_in state
        # ff out
        ff, intermediate_x = feed_forward(x_norm, self.weights_linear)
        self.ff1 = intermediate_x

        ff = ff + x

        return ff, attention_scores

    def backward(self, grad_output):
        '''
        backward pass
        '''

        # feed forward
        self.weights_linear['b2'].grad_weight += np.sum(grad_output, axis=(0,1))
        self.weights_linear['W2'].grad_weight += self.ff1.reshape(-1, self.d_ff).T @ grad_output.reshape(-1, self.d_model)
        grad = grad_output @ self.weights_linear['W2'].weight.T

        grad = grad * (self.ff1 > 0)

        self.weights_linear['b1'].grad_weight += np.sum(grad, axis=(0,1))
        self.weights_linear['W1'].grad_weight += self.normed_2.reshape(-1, self.d_model).T @ grad.reshape(-1, self.d_ff)
        grad = (grad @ self.weights_linear['W1'].weight.T)
        


        # layer norm ff
        self.weights_norm_ff['beta'].grad_weight += np.sum(grad, axis=(0,1))
        self.weights_norm_ff['gamma'].grad_weight += np.sum(grad * self.prescale2, axis=(0,1))
        
        # formula from online
        grad = grad * self.weights_norm_ff['gamma'].weight  # <--- new line
        grad = (1.0 / self.d_model) * self.inv_std2 * (
            self.d_model * grad
            - np.sum(grad, axis=-1, keepdims=True)
            - self.prescale2 * np.sum(grad * self.prescale2, axis=-1, keepdims=True)
        )

        # add residual
        grad += grad_output

        # attention
        self.weights_attention['bo'].grad_weight += np.sum(grad, axis=(0,1))
        self.weights_attention['Wo'].grad_weight += self.x_combined.reshape(-1, self.d_model).T @ grad.reshape(-1, self.d_model)
        grad = grad @ self.weights_attention['Wo'].weight.T

        grad = split_heads(grad, self.num_heads) # (B, H, L, Dh)

        grad_v = np.matmul(self.softmax_scores.transpose(0,1,3,2), grad)
        grad_scores = np.matmul(grad, self.qkv_split[2].transpose(0,1,3,2))
        grad_scores = self.softmax_scores * (grad_scores - np.sum(grad_scores * self.softmax_scores, axis=-1, keepdims=True))

        # Grad wrt Qh, Kh
        grad_q = np.matmul(grad_scores, self.qkv_split[1]) / (self.Dh ** 0.5)
        grad_k = np.matmul(grad_scores.transpose(0,1,3,2), self.qkv_split[0]) / (self.Dh ** 0.5)

        # Combine heads
        grad_q = combine_heads(grad_q)
        grad_k = combine_heads(grad_k)
        grad_v = combine_heads(grad_v)

        self.weights_attention['bv'].grad_weight += np.sum(grad_v, axis=(0,1))
        self.weights_attention['Wv'].grad_weight += self.qkv[2].reshape(-1, self.d_model).T @ grad_v.reshape(-1, self.d_model)

        self.weights_attention['bk'].grad_weight += np.sum(grad_k, axis=(0,1))
        self.weights_attention['Wk'].grad_weight += self.qkv[1].reshape(-1, self.d_model).T @ grad_k.reshape(-1, self.d_model)

        self.weights_attention['bq'].grad_weight += np.sum(grad_q, axis=(0,1))
        self.weights_attention['Wq'].grad_weight += self.qkv[0].reshape(-1, self.d_model).T @ grad_q.reshape(-1, self.d_model)

        grad = grad_q @ self.weights_attention['Wq'].weight.T + grad_k @ self.weights_attention['Wk'].weight.T + grad_v @ self.weights_attention['Wv'].weight.T



        # layer norm att
        self.weights_norm_att['beta'].grad_weight += np.sum(grad, axis=(0,1))
        self.weights_norm_att['gamma'].grad_weight += np.sum(grad * self.prescale1, axis=(0,1))
        
        # formula from online
        grad = grad * self.weights_norm_att['gamma'].weight
        grad = (1.0 / self.d_model) * self.inv_std1 * (
            self.d_model * grad
            - np.sum(grad, axis=-1, keepdims=True)
            - self.prescale1 * np.sum(grad * self.prescale1, axis=-1, keepdims=True)
        )


        return grad


class ViT(Transformer):
    '''
    vision transformer
    '''
    def __init__(self, num_heads, input_shape):
        super().__init__(num_heads, input_shape)

    
    def forward(self, x):
        return super().forward(x, x, x)[0]


class Classifier:
    '''
    simple linear classifier head
    '''
    def __init__(self, input_dim, num_classes):
        self.weights = {
            "W": param(random_init((input_dim, num_classes)), np.zeros((input_dim, num_classes)), np.zeros((input_dim, num_classes)), np.zeros((input_dim, num_classes))),
            "b": param(random_init((num_classes,)), np.zeros((num_classes,)), np.zeros((num_classes,)), np.zeros((num_classes,))),
        }
        self.weight_names = ['weights']

    def forward(self, x):
        self.x = x
        W, b = self.weights["W"].weight, self.weights["b"].weight

        out = x @ W + b

        return out

    def backward(self, grad_output):
        '''
        backward pass
        '''

        self.weights["b"].grad_weight += np.sum(grad_output, axis=0)

        self.weights["W"].grad_weight += self.x.reshape(-1, self.x.shape[-1]).T  @ grad_output.reshape(-1, grad_output.shape[-1])

        grad = grad_output @ self.weights["W"].weight.T

        return grad