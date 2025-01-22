'''
    In this document I implement the most relevant loss functions

    - L1 Loss (Mean Absolute Error)
    - MSE Loss (Mean Squared Error)
    - Cross Entropy Loss (for multi-class classification)
    - Binary Cross Entropy (BCE) and BCE with Logits
    - Negative Log Likelihood (NLL) Loss
    - Smooth L1 Loss (Huber Loss)
    - Hinge Loss / Margin Ranking Loss
    - Multi-label / Multi-class Losses (MultiLabelSoftMarginLoss)
    - KL Divergence Loss
    - Triplet Margin Loss (for Metric Learning)
    - Cosine Embedding Loss
    - Custom Loss Example
'''

import torch
import torch.nn as nn

def L1_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    '''
        predictions: shape [N, ...]
        targets: shape [N, ...]
        returns: scalar tensor (the mean of absolute difference sum(|p - t|)/len(p) )
    '''

    return torch.abs(predictions - targets).mean()

def L2_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    '''
        predictions: shape [N, ...]
        targets: shape [N, ...]
        returns: scalar tensor (the mean of squared difference sum(||p - t||**2)/len(p) )
    '''
    diff = predictions - targets
    return (diff * diff).mean()  


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    '''
        logits: shape [N, C]
        targets: shape [N]
        returns: scalar tensor (the average cross entropy over the batch)
    '''

    n = logits.shape[0]

    # log_sum_exp over classes for each sample
    # shape [N]
    log_sum_exp = torch.logsumexp(logits, dim=1)

    # Gather the logit corresponding to the correct class
    # shape [N]
    logit_correct_class = logits[torch.arange(n), targets]

    # cross entropy = logsumexp - logit of correct class
    loss = (log_sum_exp - logit_correct_class).mean()
    
    return loss

def binary_cross_entropy_loss(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    '''
        probabilities: shape [N] or [N,1], values in (0,1)
        targets: shape [N], values in {0,1 }
        returns: scalar tensor (the average binary cross entropy over the batch)

        Equation per sample: BCE(p, y) = - [y * log(p) + (1-y) * log(1-p)]
    '''
    eps = 1e-12 

    # Flatten tensor if needed
    probabilities = probabilities.view(-1)
    targets = targets.view(-1)

    loss = -(targets * torch.log(probabilities + eps) + (1.0 - targets) * torch.log(1.0 - probabilities + eps))
    return loss.mean()

def binary_cross_entropy_with_logits_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    '''
        In this function we combine the sigmoid with the cross entropy loss 
        logits: shape [N], unbounded real values
        targets: shape [N], values in {0,1 }
        returns: scalar tensor (the average binary cross entropy over the batch)

        Equation per sample: BCE(p, y) = - [y * log(p) + (1-y) * log(1-p)]
    '''
    # Flatten tensor if needed
    logits = logits.view(-1)
    targets = targets.view(-1)

    # max_val = max(logits, 0)
    max_val = torch.clamp(logits, min=0)

    loss = -(targets * torch.log(probabilities + eps) + (1.0 - targets) * torch.log(1.0 - probabilities + eps))
    return loss.mean()

if __name__ == "__main__":
    # Example usage
    logits = torch.tensor([[2.0, 1.0, 0.1],
                        [0.1, 2.1, 2.0]], requires_grad=True)  # shape [2, 3]
    labels = torch.tensor([0, 2])  # shape [2]
    loss_val = cross_entropy_loss(logits, labels)
    print("Cross Entropy Loss:", loss_val.item())

    loss_val.backward()
    print("Grad for logits:\n", logits.grad)

