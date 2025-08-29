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


def step_sgd(model, learning_rate):
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


def step_adam(model, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, t=1, max_norm=1.0):
    for name in model.weight_names:
        attr = getattr(model, name)
        for key in attr.keys():
            p = attr[key]
            # initialize m/v if missing
            if p.m is None: p.m = np.zeros_like(p.weight)
            if p.v is None: p.v = np.zeros_like(p.weight)

            # gradient clipping
            norm = np.linalg.norm(p.grad_weight)
            if norm > max_norm:
                p.grad_weight = p.grad_weight * (max_norm / norm)

            # update moments
            p.m = beta1 * p.m + (1 - beta1) * p.grad_weight
            p.v = beta2 * p.v + (1 - beta2) * (p.grad_weight ** 2)

            # bias-corrected
            m_hat = p.m / (1 - beta1**t)
            v_hat = p.v / (1 - beta2**t)

            # update weight
            p.weight -= lr * m_hat / (np.sqrt(v_hat) + eps)



def zero_grad(model):
    for name in model.weight_names:
        attr = getattr(model, name)  
       
        # for each weight name
        for key in attr.keys():
            attr[key].grad_weight[:] = 0


def random_init(shape):
    return np.random.randn(*shape) * 0.01
