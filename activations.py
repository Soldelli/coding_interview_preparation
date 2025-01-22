'''
    In this document I implement the most relevant activation functions
    integrated with the pytorch autograd class for forward/backwad compatibility

    Supported activations:
    - Sigmoid
    - Thanh
    - ReLU
    - Leaky ReLU
    - GeLU

    Future integrations:
    - SeLU
''' 

import math
import torch
from torch.autograd import Function 


class Sigmoid(Function):
    
    @staticmethod
    def forward(ctx, input):
        '''
            Equation for Sigmoid: s(x) = 1 / (1 + exp(-x))
        '''
        output = 1.0 / ( 1.0 + torch.exp(-input))
        ctx.save_for_backward(output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        '''
            Equation for derivative of Sigmoid: s'(x) = s(x)(1 - s(x))
        '''
        (output,) = ctx.saved_tensors
        grad_input = grad_output * (output * (1.0 - output))
        return grad_input
    

class Tanh(Function):
    
    @staticmethod
    def forward(ctx, input):
        '''
            Equation for Tanh: tanh(x) = ( exp(x) - exp(-x) )/( exp(x) + exp(-x) )
        '''
        # output = (torch.exp(input) - torch.exp(-input)) / (torch.exp(input) + torch.exp(-input))  # if we want to manually define the function, it can have numerica instability
        output = torch.tanh(input)  # if we want to use the 
        ctx.save_for_backward(output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        '''
            Equation for derivative of Tanh: tanh'(x) = 1 - tanh(x)**2
        '''
        (output,) = ctx.saved_tensors
        grad_input = grad_output * (1.0 - output * output)
        return grad_input
    

class ReLU(Function):
    
    @staticmethod
    def forward(ctx, input):
        '''
            Equation for ReLU: relu(x) = input if input > 0 else 0
        '''
        output = torch.clamp(input, min=0) 
        ctx.save_for_backward(output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        '''
            Equation for derivative of ReLU: relu'(x) = 1 if output > 0 else 0 
        '''
        (output,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0
        return grad_input
    

class LeakyReLU(Function):
    
    @staticmethod
    def forward(ctx, input, negative_slope=0.01):
        '''
            Equation for LeakyReLU: leaky_relu(x) = input if input > 0 else input * negative_slope 
        '''
        output = torch.where(input >= 0, input, input * negative_slope) 
        ctx.save_for_backward(output)
        ctx.negative_slope = negative_slope
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        '''torch
            Equation for derivative of LeakyReLU: leaky_relu'(x) = 1 if output > 0 else 0 
        '''
        (output,) = ctx.saved_tensors
        negative_slope = ctx.negative_slope

        grad_input = grad_output.clone()
        grad_input[output < 0] = negative_slope
        return grad_input
    

class GELU(Function):
    
    @staticmethod
    def forward(ctx, input, negative_slope=0.01):
        '''
            Equation for GELU: gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        '''
        output = 0.5 * input * (1 + torch.erf(input / math.sqrt(2.0)))
        ctx.save_for_backward(input, output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        '''
            Equation for derivative of GELU: gelu'(x) =  term1 + term2
            term 1 = 0.5 * (1 + erf(x / sqrt(2)))
            term 2 = 0.5 * x * (1 / sqrt(2 * pi)) * exp( - x **2 /2)
        '''
        input, output = ctx.saved_tensors

        term_1 = 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))
        term_2 = 0.5 * input * (1.0 / math.sqrt(2.0 * math.pi)) * torch.exp(- 0.5 * input ** 2)

        grad_input = grad_output * (term_1 + term_2)
        return grad_input