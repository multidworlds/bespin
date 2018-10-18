import numpy as np
import matplotlib.pyplot as pl

flux, sigma, time = np.loadtxt("dat/2m1324_visit6_lc.dat", unpack=True)
pl.plot(time, flux)
pl.show()
