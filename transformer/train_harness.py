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


def train_vit(model, train_data, epochs=10, learning_rate=0.001, embed_shape=(4, 65,64)):
    '''
    train function
    
    very basic, no need for anything fancy if I'm just playing
    '''
    for epoch in range(epochs):
        for batch in train_data:

            # Forward pass
            # use CLS token output for classification
            embeddings = model["embedding"].forward(batch['inputs'])
            outputs = model['transformer'].forward(embeddings)
            outputs = model['classifier'].forward(outputs[:, 0, :])  

            loss, grad = utils.compute_loss(outputs, batch['labels'])
            grad = model["classifier"].backward(grad)

            grad_transformer = np.zeros((embed_shape[0], embed_shape[1], embed_shape[2]))
            grad_transformer[:, 0, :] = grad    

            grad = model['transformer'].backward(grad_transformer)
            model["embedding"].backward(grad)

            # # Update weights
            for key in model.keys():
                utils.step(model[key], learning_rate)
                utils.zero_grad(model[key])

        print(f"Epoch {epoch + 1}/{epochs} completed.")

        print(f"  Loss: {loss}")


if __name__ == "__main__":
    # Example usage
    
    batch_size = 4      # batch size
    input_shape = (128, 128, 3)
    train_data, test_data = data.load_image_dataset("/Users/elliotweiner/software/TransformerFromScratch/transformer/data/animals", image_size=input_shape, train_split=1.0, batch_size=batch_size)

    # Dummy dataset parameters
    total_samples = len(train_data)   # total number of samples
    embed_dim = 64      # embedding dimension
    num_classes = 2    # number of classes
    
    embedding_shape = [1 + input_shape[0] // 16 * input_shape[1] // 16, embed_dim]

    num_heads = 8
    

    vit_model = {
        "embedding": models.ViT_embedding(input_shape, embed_dim),
        "transformer": models.ViT(num_heads, embedding_shape),
        "classifier": models.Classifier(embedding_shape[-1], num_classes)
    }

    print("Dataset Stats")

    train = np.array(train_data)
    test = np.array(test_data)

    print(f"  Set size:")
    print(f"    train - {train.shape} with shape {train[0]["inputs"].shape}")
    print(f"    test - {test.shape}")
    print("\nTraining...\n")

    train_vit(vit_model, train, epochs=10, learning_rate=0.01)