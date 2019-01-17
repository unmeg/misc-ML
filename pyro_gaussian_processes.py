from __future__ import print_function
import os
import matplotlib.pyplot as plt
import torch

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist

assert pyro.__version__.startswith('0.3.0')
pyro.enable_validation(True)       # can help with debugging
pyro.set_rng_seed(0)

def plot(plot_observed_data=False, plot_predictions=False, n_prior_samples=0,
         model=None, kernel=None, n_test=500):

    plt.figure(figsize=(12, 6))
    if plot_observed_data:
        plt.plot(X.numpy(), y.numpy(), 'kx')
    if plot_predictions:
        Xtest = torch.linspace(-0.5, 5.5, n_test)  # test inputs
        # compute predictive mean and variance
        with torch.no_grad():
            if type(model) == gp.models.VariationalSparseGP:
                mean, cov = model(Xtest, full_cov=True)
            else:
                mean, cov = model(Xtest, full_cov=True, noiseless=False)
        sd = cov.diag().sqrt()  # standard deviation at each input point x
        plt.plot(Xtest.numpy(), mean.numpy(), 'r', lw=2)  # plot the mean
        plt.fill_between(Xtest.numpy(),  # plot the two-sigma uncertainty about the mean
                         (mean - 2.0 * sd).numpy(),
                         (mean + 2.0 * sd).numpy(),
                         color='C0', alpha=0.3)
    if n_prior_samples > 0:  # plot samples from the GP prior
        Xtest = torch.linspace(-0.5, 5.5, n_test)  # test inputs
        noise = (model.noise if type(model) != gp.models.VariationalSparseGP
                 else model.likelihood.variance)
        cov = kernel.forward(Xtest) + noise.expand(n_test).diag()
        samples = dist.MultivariateNormal(torch.zeros(n_test), covariance_matrix=cov)\
                      .sample(sample_shape=(n_prior_samples,))
        plt.plot(Xtest.numpy(), samples.numpy().T, lw=2, alpha=0.4)

    plt.xlim(-0.5, 5.5)
    plt.show()

if __name__ == '__main__':

    N = 2500
    X = dist.Uniform(0.0, 5.0).sample(sample_shape=(N,))
    y = 0.5 * torch.sin(4*X) + dist.Normal(0.0, 0.3).sample(sample_shape=(N,))
    model_type = 3

    plot(plot_observed_data=True)

    # initialize the inducing inputs
    Xu = torch.arange(10.) / 2.0 

    # initialize the kernel
    kernel = gp.kernels.Periodic(input_dim=1)

    # define the likelihood
    likelihood = gp.likelihoods.Gaussian() # TODO: check if correct initial likelihood?

    # define the model
    # note: "whiten" flag enables more stable optimization
    vsgp = gp.models.VariationalSparseGP(X, y, kernel, Xu=Xu, likelihood=likelihood, whiten=True)
    sgpr = gp.models.SparseGPRegression(X, y, kernel, Xu=Xu, jitter=1.0e-5)

    if model_type == 1:
        optimizer = torch.optim.Adam(sgpr.parameters(), lr=0.005)
        loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        losses = []
        num_steps = 2500
        for i in range(num_steps):
            optimizer.zero_grad()
            loss = loss_fn(sgpr.model, sgpr.guide)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        plt.plot(losses)
        plt.show()
        plot(model=sgpr, plot_observed_data=True, plot_predictions=True)
    if model_type == 2:
        # the GP module has its own training loop management, not sure if better than/equal to custom
        num_steps = 2500
        losses = gp.util.train(vsgp, num_steps=num_steps)
        plt.plot(losses)
        plt.show()
        plot(model=vsgp, plot_observed_data=True, plot_predictions=True)
    if model_type == 3:
        optimizer = torch.optim.Adam(vsgp.parameters(), lr=0.005)
        loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        losses = []
        num_steps = 2500
        for i in range(num_steps):
            optimizer.zero_grad()
            loss = loss_fn(vsgp.model, vsgp.guide)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        plt.plot(losses)
        plt.show()
        plot(model=vsgp, plot_observed_data=True, plot_predictions=True)
