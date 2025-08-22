import numpy as np

# cross entropy loss
def compute_loss(logits, labels):
    # softmax
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # cross-entropy
    log_likelihood = -np.sum(labels * np.log(probs + 1e-9), axis=1)
    loss = np.mean(log_likelihood)

    return loss


def step(model, learning_rate):
    '''
    each param needs a grad attribute
    simple SGD step
    '''
    for param in model.weights_attention.keys():
        model.weights_attention[param].weight -= learning_rate * model.weights_attention[param].grad_weight

    for param in model.weights_linear.keys():
        model.weights_linear[param].weight -= learning_rate * model.weights_linear[param].grad_weight


def zero_grad(model):
    for param in model.weights_attention.keys():
        model.weights_attention[param].grad_weight[:] = 0.0

    for param in model.weights_linear.keys():
        model.weights_linear[param].grad_weight[:] = 0.0


def random_init(shape):
    return np.random.randn(*shape) * 0.01
