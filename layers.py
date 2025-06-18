import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchNormalization(nn.Module):

    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        super(BatchNormalization, self).__init__()

        self.num_features = num_features

        # Learnable parameters: scale (gamma) and shift (beta)
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.ones(num_features))
        
        # Running statistics
        self.running_mean = torch.zeros(num_features)
        self.running_variance = torch.ones(num_features)

        # momentum and epsilon
        self.momentum = momentum
        self.eps = eps

    def forward(self, x):
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0)
            x_normalized = (x - batch_mean) / torch.sqrt(batch_var + self.eps)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_variance = self.momentum * self.running_variance + (1 - self.momentum) * batch_var

        else:
            x_normalized = (x - self.running_mean) / torch.sqrt(self.running_variance + self.eps)

        out = x_normalized * self.gamma + self.beta
        return out
    
    def backward(self, grad_output, x):
        if self.training:
            # recompute batch mean and variance and normalize input: 

            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0)
            x_normalized = (x - batch_mean) / torch.sqrt(batch_var + self.eps)

            # compute gradients with respect to gamma and beta:
            grad_gamma = torch.sum(grad_output * x_normalized, dim=0)
            grad_beta = torch.sum(grad_output, dim=0)

            # compute gradient with respect to input
            N = x.size

            grad_x_normalized = grad_output * self.gamma
            grad_var = torch.sum(grad_x_normalized * (x - batch_mean) * -0.5 * (batch_var + self.eps)**(-1.5), dim=0)
            grad_mean = torch.sum(grad_x_normalized * -1.0 / torch.sqrt(batch_var + self.eps), dim=0) + grad_var * torch.mean(-2.0 * (x - batch_mean), dim=0)

            grad_x = grad_x_normalized / torch.sqrt(batch_var + self.eps) + grad_var * 2.0 * (x - batch_mean) / N + grad_mean / N

            return grad_x, grad_gamma, grad_beta


        else:
            raise NotImplementedError('Not implemented for inference time')
        