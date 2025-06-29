import torch
import torch.nn as nn


# GRADIENT REVERSAL LAYER IMPLEMENTATION

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer implementation for domain adversarial training.
    During forward pass, acts as identity. During backward pass, multiplies
    gradients by -lambda (domain adaptation factor).
    """
    @staticmethod
    def forward(ctx, x, lambda_param):
        ctx.lambda_param = lambda_param
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_param
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_param=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_param = lambda_param
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_param)
    
    def set_lambda(self, lambda_param):
        self.lambda_param = lambda_param