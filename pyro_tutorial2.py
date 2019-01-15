import matplotlib.pyplot as plt
import numpy as np
import torch

import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist

pyro.set_rng_seed(101)


# weight given guess is sampled from a normal distro with mean: guess, cov: 1
# measurement | guess and weight is " ... " with mean: weight, cov: 0.75 (slightly less)
# so we're saying the relationship between the weight and the guess is strong
# the weight between the guess and weight and the measurement is slightly less strong

def scale(guess):
    weight = pyro.sample("weight", dist.Normal(guess, 1.0))
    return pyro.sample("measurement", dist.Normal(weight, 0.75))

# it is possible to condition on observed data
# we can use pyro.condition to constrain the value sof sample statements
# takes a model and dictionary of observations
# returns a new model that has the same input and output signatures but uses the given values at observated sample statements

conditioned_scale = pyro.condition(scale, data={"measurement": 9.5})

def deferred_conditioned_scale(measurement, *args, **kwargs):
    return pyro.condition(scale, data={"measurement": measurement})(*args, **kwargs)

def scale_obs(guess):
    weight = pyro.sample("weight", dist.Normal(guess, 1.))
    return pyro.sample("measurement", dist.Normal(weight, 1.), obs=9.5) # conditions on the measurement 9.5

    # pyro.condition and pyro.do (an operator for causal inference copied from Pearl) can be mixed and composed freely

## GUIDE FUNCTIONS
# inference algorithms allow us to use stochastic functions as approximate posterior distributions

# guide functions must satisfy two criteria to be valid approximations:
# 1) all unobserved (i.e. not conditioned) sample statements that appear in the model appear in the guide
# 2) the guide has the same input signature (input arguments) as the model

# guide functions are programmable, data-dependent proposal distributions for importance sample, rejection sampling, sequential 
# monte carlo, MCMC and independent metropolis-hastings

# MCMC, importance sampling and stochastic variational inference are currently implemented in pyro

# guide function plays a different role in the different algorithms but broadly it should be chosen so that it is 
# flexible enough to closely approximate the distro over all unobserved sample statements in the model

def perfect_guide(guess):
    loc = (0.75**2 * guess + 9.5) / (1 + 0.75**2)
    scale = np.sqrt(0.75**2/(1+0.75**2))
    return pyro.sample("weight", dist.Normal(loc,scale))


print(perfect_guide(8.5))