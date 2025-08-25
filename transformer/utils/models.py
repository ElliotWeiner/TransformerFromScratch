from .layers import *
from .utils import random_init

from dataclasses import dataclass
import numpy as np
        
@dataclass
class param:
    weight: np.ndarray
    grad_weight: np.ndarray


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
            "W": param(random_init((patch_dim, embed_dim)), np.zeros((patch_dim, embed_dim))),
            "b": param(random_init((embed_dim,)), np.zeros((embed_dim,))),
        }
        
        self.H, self.W, self.D = self.input_shape
        self.num_patches_h = self.H // self.patch_size
        self.num_patches_w = self.W // self.patch_size
        self.L = self.num_patches_h * self.num_patches_w  # total patches

        self.input_patch_size = self.patch_size * self.patch_size * self.D

        # add CLS token
        self.cls_token = param(random_init((1, 1, embed_dim)), np.zeros((1, 1, embed_dim)))

        # add position embedding
        self.pos_embedding = param(random_init((1, self.L + 1, embed_dim)), np.zeros((1, self.L + 1, embed_dim)))

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

        out = np.concatenate([np.tile(self.cls_token.weight, (B, 1, 1)), out], axis=1)
        out += self.pos_embedding.weight
        

        return out

    # TODO
    def backward(self, grad_output):
        '''
        backward pass
        '''
        self.pos_embedding.grad_weight += grad_output

        self.cls_token.grad_weight += np.sum(grad_output[:,0,:], axis=0)

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

        self.weights_attention = {
            "Wq": param(random_init((input_shape[-1], input_shape[-1])), np.zeros((input_shape[-1], input_shape[-1]))),
            "bq": param(random_init((input_shape[-1],)), np.zeros((input_shape[-1],))),
            
            "Wk": param(random_init((input_shape[-1], input_shape[-1])), np.zeros((input_shape[-1], input_shape[-1]))),
            "bk": param(random_init((input_shape[-1],)), np.zeros((input_shape[-1],))),
            
            "Wv": param(random_init((input_shape[-1], input_shape[-1])), np.zeros((input_shape[-1], input_shape[-1]))),
            "bv": param(random_init((input_shape[-1],)), np.zeros((input_shape[-1],))),
            
            "Wo": param(random_init((input_shape[-1], input_shape[-1])), np.zeros((input_shape[-1], input_shape[-1]))),
            "bo": param(random_init((input_shape[-1],)), np.zeros((input_shape[-1],))),
        }

        self.d_model = input_shape[-1]
        self.d_ff = 4 * self.d_model

        self.weights_linear = {
            "W1": param(random_init((self.d_model, self.d_ff)), np.zeros((self.d_model, self.d_ff))),
            "b1": param(random_init((self.d_ff,)), np.zeros((self.d_ff,))),
            "W2": param(random_init((self.d_ff, self.d_model)), np.zeros((self.d_ff, self.d_model))),
            "b2": param(random_init((self.d_model,)), np.zeros((self.d_model,))),
        }

        self.weights_norm_att = {
            "gamma": param(random_init((self.d_model,)), np.ones((self.d_model,))),
            "beta": param(random_init((self.d_model,)), np.zeros((self.d_model,))),
        }

        self.weights_norm_ff = {
            "gamma": param(random_init((self.d_model,)), np.ones((self.d_model,))),
            "beta": param(random_init((self.d_model,)), np.zeros((self.d_model,))),
        }

        self.weight_names = ['weights_attention', 'weights_linear', 'weights_norm_att', 'weights_norm_ff']

    def forward(self, Q, K, V):
        '''
        forward pass
        '''
        self.x = (Q, K, V)

        x_norm = layer_norm(x, self.weights_norm_att)
        self.n1 = x_norm

        x, attention_scores = self_attention(x_norm, self.num_heads, self.weights_attention)
        self.mha = x

        x = Q + x


        x_norm = layer_norm(x, self.weights_norm_ff)
        self.n2 = x_norm
        ff, intermediate_x = feed_forward(x_norm, self.weights_linear, self.weights_norm_ff)
        self.ff1 = intermediate_x
        self.ff2 = ff

        ff = ff + x

        return ff

    # TODO
    def backward(self, grad_output):
        '''
        backward pass
        '''

        # feed forward
        self.weights_linear['b2'].grad_weight += np.sum(grad_output, axis=0)
        self.weights_linear['W2'].grad_weight += self.ff1.reshape(-1, self.d_ff).T @ grad_output.reshape(-1, self.d_model)
        grad = grad_output @ self.weights_linear['W2'].weight.T

        self.weights_linear['b1'].grad_weight += np.sum(grad * (self.ff1 > 0), axis=0)
        self.weights_linear['W1'].grad_weight += self.n2.reshape(-1, self.d_model).T @ grad.reshape(-1, self.d_ff)
        grad = (grad @ self.weights_linear['W1'].weight.T) * (self.ff1 > 0)
        

        # layer norm ff
        self.weights_norm_ff['beta'].grad_weight += np.sum(grad, axis=(0,1))
        self.weights_norm_ff['gamma'].grad_weight += 
        grad = 


        # attention
        self.weights_attention['bo'].grad_weight += np.sum(grad, axis=0)
        self.weights_attention['Wo'].grad_weight += 
        grad = 

        self.weights_attention['bv'].grad_weight += np.sum(grad, axis=0)
        self.weights_attention['Wv'].grad_weight += 
        grad =

        self.weights_attention['bk'].grad_weight += np.sum(grad, axis=0)
        self.weights_attention['Wk'].grad_weight += 
        grad =

        self.weights_attention['bq'].grad_weight += np.sum(grad, axis=0)
        self.weights_attention['Wq'].grad_weight += 
        grad =


        # layer norm att
        self.weights_norm_att['beta'].grad_weight += np.sum(grad, axis=(0,1))
        self.weights_norm_att['gamma'].grad_weight += 
        grad = 


        return grad


# shouldn't event need to touch
class ViT(Transformer):
    '''
    vision transformer
    '''
    def __init__(self, num_heads, input_shape):
        super().__init__(num_heads, input_shape)

    
    def forward(self, x):
        return super().forward(x, x, x)


