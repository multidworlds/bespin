import starry
import matplotlib.pyplot as pl
import numpy as np
import george
from george import kernels
np.random.seed(42)


def power(l):
    """A very simple power spectrum."""
    return l ** -2


# Let's draw coefficients at random
# according to the simple power spectrum
lmax = 10
y = np.array([1.])
for l in range(1, lmax + 1):
    p = power(l)
    yl = np.random.randn(2 * l + 1)
    yl *= np.sqrt(p / np.sum(yl ** 2))
    y = np.concatenate((y, yl))

# Add variability
ntime = 100
time = np.linspace(0, 10, ntime)
amp = 1.0
tau = 1.0
kernel = [None] + [amp * power(l) ** 2 * kernels.ExpSquaredKernel(tau ** 2)
                   for l in range(1, lmax + 1)]
gp = [None] + [george.GP(kernel[l]) for l in range(1, lmax + 1)]
y_of_t = np.concatenate([[gp[l].sample(time) for m in range(-l, l + 1)] for l in range(1, lmax + 1)])
y_of_t = np.concatenate((np.ones_like(time).reshape(1, -1), y_of_t))

# Instantiate and plot the map
map = starry.Map(lmax, ntime)
map[:, :] = y.reshape(-1, 1) + y_of_t
map.show(gif="clouds.gif", show_labels=False)
