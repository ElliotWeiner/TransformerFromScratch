'''

We will attempt to build a ViT (Vision Transformer) using my defined transformer model.

'''

from utils import models, data, utils


def train_vit(model, train_data, epochs=10, learning_rate=0.001):
    '''
    train function
    
    very basic, no need for anything fancy if I'm just playing
    '''
    for epoch in range(epochs):
        for batch in train_data:
            # Forward pass
            outputs = model.forward(batch)
            loss = utils.compute_loss(outputs, batch.labels)
            

            # Backward pass
            model.backward(loss)

            # Update weights
            utils.step(model, learning_rate)
            model.zero_grad()

        print(f"Epoch {epoch + 1}/{epochs} completed.")

        print(f"  Loss: {loss.item()}")


if __name__ == "__main__":
    # Example usage
    num_heads = 8
    input_shape = (224, 224, 3)

    vit_model = models.ViT(num_heads, input_shape)
    train_data = data.load_data()

    train_vit(vit_model, train_data, epochs=10, learning_rate=0.001)