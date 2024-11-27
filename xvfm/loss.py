import torch
from torch.autograd import Function

class SSMGaussian(Function):
    
    @staticmethod
    def forward(ctx, posterior, x):
        nll = -posterior.log_prob(x)
        ctx.save_for_backward(x, posterior.mean, posterior.sigma)
        return nll.mean()

    @staticmethod
    def backward(ctx, grad_output):
        x, mu, sigma = ctx.saved_tensors
        diff = x - mu
        
        # Empirical and expected sufficient statistics
        empirical_mu = x.mean(dim=0)
        expected_mu = mu.mean(dim=0)

        empirical_sigma = torch.einsum('bi,bj->bij', diff, diff).mean(dim=0)
        expected_sigma = sigma.mean(dim=0)

        # Compute difference in sufficient statistics
        diff_mu = empirical_mu - expected_mu
        diff_sigma = empirical_sigma - expected_sigma

        # Compute Jacobians of the natural parameters w.r.t the neural network parameters
        grad_mu_jacobian = torch.autograd.grad(mu, [p for p in mu], grad_outputs=diff_mu, retain_graph=True)
        grad_sigma_jacobian = torch.autograd.grad(sigma, [p for p in sigma], grad_outputs=diff_sigma, retain_graph=True)

        # Scale differences by Jacobians
        grad_mu_scaled = torch.cat([p.grad.view(-1) for p in grad_mu_jacobian]).sum()
        grad_sigma_scaled = torch.cat([p.grad.view(-1) for p in grad_sigma_jacobian]).sum()

        return None, grad_mu_scaled, grad_sigma_scaled


class SSMNatGradGaussian(Function):
    @staticmethod
    def forward(ctx, posterior, x):
        nll = -posterior.log_prob(x)
        ctx.save_for_backward(x, posterior.mean, posterior.sigma)
        return nll.mean()

    @staticmethod
    def backward(ctx, grad_output):
        x, mu, sigma = ctx.saved_tensors
        inv_sigma = torch.inverse(sigma)
        diff = x - mu
        
        # Empirical and expected sufficient statistics
        empirical_mu = x.mean(dim=0)
        expected_mu = mu.mean(dim=0)

        empirical_sigma = torch.einsum('bi,bj->bij', diff, diff).mean(dim=0)
        expected_sigma = sigma.mean(dim=0)

        # Compute difference in sufficient statistics
        diff_mu = empirical_mu - expected_mu
        diff_sigma = empirical_sigma - expected_sigma

        # Fisher Information Matrix
        fisher_mu = inv_sigma  # Correct FIM for \mu
        fisher_sigma = 0.5 * torch.kron(inv_sigma, inv_sigma)  # Correct FIM for \Sigma

        # Natural gradients
        natural_grad_mu = torch.einsum('ij,j->i', fisher_mu, diff_mu)
        natural_grad_sigma = torch.einsum('ij,jk->ik', fisher_sigma, diff_sigma.flatten()).view_as(sigma)

        return None, natural_grad_mu, natural_grad_sigma