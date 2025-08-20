from . import layers

class Transformer:
    '''
    transformer model
    '''

    def __init__(self, num_heads):
        self.num_heads = num_heads
        self.weights = [None, None]
        self.biases = [None, None]

    def forward(self, Q, K, V):
        '''
        forward pass
        '''
        return layers.transformer_block(Q, K, V, self.num_heads, self.weights, self.biases)
    
    def backward(self, grad_output):
        '''
        backward pass
        '''
        pass
