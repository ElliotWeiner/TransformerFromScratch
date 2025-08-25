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
        self.weight_names = ['weights']

        self.H, self.W, self.D = self.input_shape
        self.num_patches_h = self.H // self.patch_size
        self.num_patches_w = self.W // self.patch_size
        self.L = self.num_patches_h * self.num_patches_w  # total patches

        # add CLS token
        self.cls_token = param(random_init((1, 1, embed_dim)), np.zeros((1, 1, embed_dim)))

        # add position embedding
        self.pos_embedding = param(random_init((1, self.L + 1, embed_dim)), np.zeros((1, self.L + 1, embed_dim)))


    def forward(self, x):
        '''
        forward pass
        '''

        B = x.shape[0]

        # patchify to 16x16
        x = x.reshape(B, self.num_patches_h, self.patch_size, self.num_patches_w, self.patch_size, self.D)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        # flatten patches to be (B, L, patch_size * patch_size * D)
        x = x.reshape(B, self.num_patches_h * self.num_patches_w, self.patch_size * self.patch_size * self.D)

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
        pass


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

        # what would this weight shape be?
        d_model = input_shape[-1]
        d_ff = 4 * d_model

        self.weights_linear = {
            "W1": param(random_init((d_model, d_ff)), np.zeros((d_model, d_ff))),
            "b1": param(random_init((d_ff,)), np.zeros((d_ff,))),
            "W2": param(random_init((d_ff, d_model)), np.zeros((d_ff, d_model))),
            "b2": param(random_init((d_model,)), np.zeros((d_model,))),
        }

        self.weights_norm = {
            "gamma": param(random_init((d_model,)), np.ones((d_model,))),
            "beta": param(random_init((d_model,)), np.zeros((d_model,))),
        }

        self.weight_names = ['weights_attention', 'weights_linear', 'weights_norm']

    def forward(self, Q, K, V):
        '''
        forward pass
        '''
        return transformer_block(Q, K, V, self.num_heads, self.weights_attention, self.weights_linear, self.weights_norm)
    
    # TODO
    def backward(self, grad_output):
        '''
        backward pass
        '''
        pass


# shouldn't event need to touch
class ViT(Transformer):
    '''
    vision transformer
    '''
    def __init__(self, num_heads, input_shape):
        super().__init__(num_heads, input_shape)

    
    def forward(self, x):
        return super().forward(x, x, x)


