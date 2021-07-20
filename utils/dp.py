
import random
import numpy as np



def laplace(mean, sensitivity, epsilon): # mean : value to be randomized (mean)
    scale = sensitivity / epsilon
    rand = random.uniform(0 ,1) - 0.5 # rand : uniform random variable
    return mean - scale * np.sign(rand) * np.log(1 - 2 * np.abs(rand))


def generate_laplace_noise(weights_shape, sensitivity, epsilon):
    mean = 0
    weights_dp_noise = np.zeros(weights_shape)
    for idx, _ in np.ndenumerate(weights_dp_noise):
        dp_noise = laplace(mean=mean, sensitivity=sensitivity, epsilon=epsilon)
        weights_dp_noise[idx] = dp_noise
    return weights_dp_noise


def gamma_noise(weights_shape, sensitivity, epsilon, num_clients):
    scale = sensitivity / epsilon
    weights_dp_noise = np.zeros(weights_shape)
    for idx, _ in np.ndenumerate(weights_dp_noise):
        dp_noise = random.gammavariate(1 / num_clients, scale) - random.gammavariate(1 / num_clients, scale)
        weights_dp_noise[idx] = dp_noise
    return weights_dp_noise

