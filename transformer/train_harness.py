'''

We will attempt to build a ViT (Vision Transformer) using my defined transformer model.

TODO:
- ADD CLS TOKEN AND CLASSIFICATION HEAD
- ADD POSITIONAL ENCODING
- ADD REAL DATA
- FIX BACKPROP

'''

from utils import models, data, utils
import numpy as np


def train_vit(model, train_data, epochs=10, learning_rate=0.001):
    '''
    train function
    
    very basic, no need for anything fancy if I'm just playing
    '''
    for epoch in range(epochs):
        for batch in train_data:
            # Forward pass
            outputs = model.forward(batch['inputs'])
            
            #loss = utils.compute_loss(outputs, batch['labels'])
            #model.backward(loss)

            # Update weights
            utils.step(model, learning_rate)
            utils.zero_grad(model)

        print(f"Epoch {epoch + 1}/{epochs} completed.")

        #print(f"  Loss: {loss}")


if __name__ == "__main__":
    # Example usage
    
    #train_data = data.load_data()

    # Dummy dataset parameters
    total_samples = 800   # total number of samples
    seq_len = 16        # sequence length (e.g., number of patches)
    embed_dim = 64      # embedding dimension
    num_classes = 2    # number of classes
    batch_size = 4      # batch size

    # Random inputs
    dummy_inputs = np.random.randn(total_samples, seq_len, embed_dim)

    # One-hot labels
    dummy_labels = np.zeros((total_samples, num_classes))
    dummy_labels[np.arange(total_samples), np.random.randint(0, num_classes, total_samples)] = 1

    # Create batches as a list of dicts
    train_data = [
        {"inputs": dummy_inputs[i:i+batch_size], "labels": dummy_labels[i:i+batch_size]}
        for i in range(0, total_samples, batch_size)
    ]

    num_heads = 8
    input_shape = (seq_len, embed_dim)

    vit_model = models.ViT(num_heads, input_shape)

    train_vit(vit_model, train_data, epochs=10, learning_rate=0.001)