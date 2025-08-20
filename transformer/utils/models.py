from . import layers

class Transformer:
    '''
    transformer model
    '''

    def __init__(self, num_heads, input_shape):
        self.num_heads = num_heads

        # TODO: initialize weights and biases correctly
        self.weights = [None, None]
        self.biases = [None, None]

    def forward(self, Q, K, V):
        '''
        forward pass
        '''
        return layers.transformer_block(Q, K, V, self.num_heads, self.weights, self.biases)
    
    # TODO
    def backward(self, grad_output):
        '''
        backward pass
        '''
        pass

    # TODO
    def zero_grad(self):
        pass




# shouldn't event need to touch
class ViT(Transformer):
    '''
    vision transformer
    '''
    def __init__(self, num_heads, input_shape):
        super().__init__(num_heads, input_shape)

    
    def forward(self, x):
        return self.forward(x, x, x)
    

