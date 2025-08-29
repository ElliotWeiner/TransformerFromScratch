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
        learning_rate = learning_rate * (0.9975 ** epoch)


        for batch in train_data:

            b = data.augment_batch(batch)

            # Forward pass
            # use CLS token output for classification
            embeddings = model["embedding"].forward(b['inputs'])
            outputs = model['transformer1'].forward(embeddings)
            outputs = model['transformer2'].forward(outputs)
            #outputs = model['transformer3'].forward(outputs)
            outputs = model['classifier'].forward(outputs[:, 0, :])  

            loss, grad = utils.compute_loss(outputs, b['labels'])
            grad = model["classifier"].backward(grad)

            grad_transformer = np.zeros((embed_shape[0], embed_shape[1], embed_shape[2]))
            grad_transformer[:, 0, :] = grad    

            #grad = model['transformer3'].backward(grad_transformer)
            grad = model['transformer2'].backward(grad_transformer)
            grad = model['transformer1'].backward(grad)
            model["embedding"].backward(grad)

            # # Update weights
            for key in model.keys():
                utils.step_adam(model[key], learning_rate)
                utils.zero_grad(model[key])

        print(f"Epoch {epoch + 1}/{epochs} completed. lr = {learning_rate}")

        print(f"  Loss: {loss}")


if __name__ == "__main__":
    # Example usage
    
    batch_size = 20      # batch size
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
        "transformer1": models.ViT(num_heads, embedding_shape),
        "transformer2": models.ViT(num_heads, embedding_shape),
        #"transformer3": models.ViT(num_heads, embedding_shape),
        "classifier": models.Classifier(embedding_shape[-1], num_classes)
    }

    print("Dataset Stats")

    train = np.array(train_data)
    test = np.array(test_data)

    print(f"  Set size:")
    print(f"    train - {train.shape} with shape {train[0]["inputs"].shape}")
    print(f"    test - {test.shape}")
    print("\nTraining...\n")



    train_vit(vit_model, train, epochs=50, learning_rate=0.0002, embed_shape=(batch_size, embedding_shape[0], embedding_shape[1]))