import torch

class SSM(Function):
    @staticmethod
    def forward(ctx, x, mu, var):
        ctx.save_for_backward(x, mu, var)
        loss = 0.5 * ((x - mu) ** 2 / var + torch.log(var))
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        x, mu, var = ctx.saved_tensors
        grad_mu = mu - x
        grad_sigma2 = (0.5 * ((x - mu) ** 2 - var)) / (var ** 2)
        return None, grad_mu, grad_sigma2
    
class SSMNG(Function):
    @staticmethod
    def forward(ctx, x, mu, var):
        ctx.save_for_backward(x, mu, var)
        loss = 0.5 * ((x - mu) ** 2 / var + torch.log(var))
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        x, mu, var = ctx.saved_tensors
        grad_mu = mu - x
        grad_sigma2 = (0.5 * ((x - mu) ** 2 - var)) / (var ** 2)
        fisher_mu = 1 / var
        fisher_sigma2 = 1 / (2 * var ** 2)
        natural_grad_mu = grad_mu / fisher_mu
        natural_grad_sigma2 = grad_sigma2 / fisher_sigma2
        return None, natural_grad_mu, natural_grad_sigma2