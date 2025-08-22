from .layers import *
from .utils import random_init

from dataclasses import dataclass
import numpy as np
        
@dataclass
class param:
    weight: np.ndarray
    grad_weight: np.ndarray

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

    def forward(self, Q, K, V):
        '''
        forward pass
        '''
        return transformer_block(Q, K, V, self.num_heads, self.weights_attention, self.weights_linear)
    
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


