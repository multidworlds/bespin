import starry
import matplotlib.pyplot as pl
import numpy as np
np.random.seed(42)


def power(l):
    """A very simple power spectrum."""
    return l ** -2


# Instantiate a map
lmax = 10
map = starry.Map(lmax=lmax)

# Now let's draw coefficients at random
# according to the simple power spectrum
y = np.array([1.])
for l in range(1, lmax + 1):
    p = power(l)
    yl = np.random.randn(2 * l + 1)
    yl *= np.sqrt(p / np.sum(yl ** 2))
    y = np.concatenate((y, yl))

# Assign the vector to the `starry` map
map[:, :] = y

# Show it
map.animate()
