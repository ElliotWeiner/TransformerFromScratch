import numpy as np

# cross entropy loss
def compute_loss(logits, labels):
    # softmax
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # cross-entropy loss
    log_likelihood = -np.sum(labels * np.log(probs + 1e-9), axis=1)  # (B,)
    loss = np.mean(log_likelihood)

    # gradient of loss w.r.t. logits
    grad_logits = (probs - labels) / logits.shape[0]  # (B, C)

    return loss, grad_logits


def step(model, learning_rate):
    '''
    each param needs a grad attribute
    simple SGD step
    '''

    # for each attribute
    for name in model.weight_names:
        attr = getattr(model, name)  
       
        # for each weight name
        for key in attr.keys():

            attr[key].weight -= learning_rate * attr[key].grad_weight


def zero_grad(model):
    for name in model.weight_names:
        attr = getattr(model, name)  
       
        # for each weight name
        for key in attr.keys():
            attr[key].grad_weight[:] = 0


def random_init(shape):
    return np.random.randn(*shape) * 0.01
