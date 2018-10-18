import starry
import matplotlib.pyplot as pl
import matplotlib.animation as animation
import numpy as np
import george
from george import kernels
from tqdm import tqdm
np.random.seed(42)
cmap = pl.get_cmap("inferno")


def power(l, amp=0.1, alpha=2.0, beta=-2.0, l0=2):
    """A broken power law power spectrum."""
    l0 = int(l0)
    if hasattr(l, "__len__"):
        p = amp * ((l + 1.0) / (l0 + 1.0)) ** alpha
        p[l >= l0] = amp * ((l[l >= l0] + 1.0) / (l0 + 1.0)) ** beta
        p[0] = 1.0
        return p
    else:
        if l == 0:
            return 1.0
        elif l < l0:
            return amp * ((l + 1.0) / (l0 + 1)) ** alpha
        else:
            return amp * ((l + 1.0) / (l0 + 1)) ** beta


def get_power(y):
    """Get the power spectrum from a map."""
    N = len(y)
    lmax = int(np.sqrt(N) - 1)
    p = [y[0] ** 2]
    n = 1
    for l in range(1, lmax + 1):
        p.append(np.sum(y[n:n + 2 * l + 1] ** 2))
        n += 2 * l + 1
    return p


def variable_map(lmax=10, amp=0.1, tau=0.025, npts=300,
                 alpha=2.0, beta=-2.0, l0=2):
    """Generate variability using a GP."""
    time = np.linspace(0.0, 1.0, npts)
    y = np.zeros(((lmax + 1) ** 2, npts))
    y[0, :] = 1.0
    n = 1
    for l in range(1, lmax + 1):
        if hasattr(tau, "__len__"):
            kernel = (power(l, amp=amp, alpha=alpha, beta=beta, l0=l0)) ** 2 * \
                     kernels.Matern32Kernel([tau[0] ** 2,
                        (tau[1] * (2 * l + 1)) ** 2], ndim=2)
            x = np.array([[ti, m] for ti in time for m in range(-l, l + 1)])
            gp = george.GP(kernel)
            sample = gp.sample(x).reshape(-1, 2 * l + 1).transpose()
        else:
            kernel = (power(l, amp=amp, alpha=alpha, beta=beta, l0=l0)) ** 2 * \
                     kernels.Matern32Kernel(tau ** 2)
            gp = george.GP(kernel)
            sample = np.array([gp.sample(time) for m in range(-l, l + 1)])
        y[n:n + 2 * l + 1, :] += sample
        n += 2 * l + 1
    return y


def plot_power(y):
    """Show the power spectrum evolution."""
    N, npts = y.shape
    lmax = int(np.sqrt(N) - 1)
    fig = pl.figure()
    ax = pl.axes((0.1, 0.1, 0.7, 0.8))
    l = np.arange(lmax + 1)
    for i in range(npts):
        ax.plot(l, get_power(y[:, i]), color=cmap(i / npts), alpha=0.3)
    ax.set_yscale("log")
    ax.set_xlabel("Spherical harmonic degree")
    ax.set_ylabel("Power")
    cax = pl.axes((0.85, 0.1, 0.05, 0.8))
    cax.imshow(np.linspace(1, 0, 1000).reshape(-1, 1),
               cmap=cmap, extent=(0, 1, 0, 1))
    cax.set_aspect('auto')
    cax.yaxis.tick_right()
    cax.set_xticks([])
    cax.set_ylabel("Time", rotation=-90)
    cax.yaxis.set_label_position("right")
    return fig, ax


def plot_coeffs(y):
    """Show the coefficient evolution."""
    N = len(y)
    lmax = int(np.sqrt(N) - 1)
    fig, ax = pl.subplots(lmax, figsize=(5, 8))
    fig.subplots_adjust(top=0.95, bottom=0.05)
    n = 1
    for l in range(1, lmax + 1):
        ymax = 1e-10
        for m in range(-l, l + 1):
            ax[l - 1].plot(y[n, :], lw=1, alpha=0.95)
            ymax = max(ymax, np.max(np.abs(y[n, :])))
            n += 1
        ax[l - 1].set_xticks([])
        ax[l - 1].set_yticks([])
        for line in ["top", "right", "bottom", "left"]:
            ax[l - 1].spines[line].set_visible(False)
        ax[l - 1].set_ylabel(l, rotation=0)
        ax[l - 1].set_ylim(-1.1 * ymax, 1.1 * ymax)
    ax[lmax - 1].set_xlabel("Time")
    ax[0].set_title("Power")
    return fig, ax


def plot_phase_curve(y, nrot=0.5):
    """Plot the map's rotational phase curve."""
    N, npts = y.shape
    lmax = int(np.sqrt(N) - 1)
    map = starry.Map(lmax)
    theta = np.linspace(0, 360 * nrot, npts)
    flux = np.zeros(npts)
    fig, ax = pl.subplots(1, figsize=(8, 5))
    for t in range(npts):
        map[:, :] = y[:, t]
        map.rotate(theta[t])
        flux[t] = map.flux()
    ax.plot(np.linspace(0, nrot, npts), flux)
    ax.set_xlabel("Phase", fontsize=14)
    ax.set_ylabel("Flux", fontsize=14)
    return fig, ax


def animate(y, res=150, cmap="plasma", gif="", interval=75, nrot=0.5):
    """Animate the map as it rotates."""
    N, npts = y.shape
    lmax = int(np.sqrt(N) - 1)
    map = starry.Map(lmax)
    I = np.zeros((npts, res, res))
    theta = np.linspace(0, 360 * nrot, npts)
    x = np.linspace(-1, 1, res)
    for t in tqdm(range(npts)):
        map[:, :] = y[:, t]
        map.rotate(theta[t])
        for i in range(res):
            for j in range(res):
                I[t, j, i] = map(x=x[i], y=x[j])
    fig, ax = pl.subplots(1, figsize=(3, 3))
    img = ax.imshow(I[0], origin="lower", interpolation="none", cmap=cmap,
                    extent=(-1, 1, -1, 1), animated=True,
                    vmin=np.nanmin(I), vmax=np.nanmax(I))
    ax.axis('off')

    def updatefig(i):
        img.set_array(I[i])
        return img,

    ani = animation.FuncAnimation(fig, updatefig, interval=interval, blit=True,
                                  frames=len(I))
    if gif != "":
        if gif.endswith(".gif"):
            gif = gif[:-4]
        ani.save('%s.gif' % gif, writer='imagemagick')
    else:
        pl.show()
    pl.close()


def example():
    """An example of a variable map."""
    y = variable_map(lmax=10, amp=0.1, tau=0.1, npts=300,
                     alpha=2.0, beta=-2.0, l0=2)
    plot_power(y)
    plot_coeffs(y)
    plot_phase_curve(y)
    animate(y, nrot=0.5)


def phase_curve(map, time, per):
    """Generate a rotational phase curve."""
    theta = 360.0 / per * time
    flux = map.flux(theta=theta)
    return np.array(np.diag(flux))


def lnlike(x, time, flux, err, map):
    """Log-likelihood."""
    # Generate the light curve from the params
    amp, tau, alpha, beta, l0, per = x[:6]
    map[:, :] = x[6:].reshape(-1, len(time))
    model = phase_curve(map, time, per)

    # Hard bounds
    if (amp < 0) or (tau < 0) or (alpha < 0) or (beta > 0) or (per < 0):
        return -np.inf
    elif (l0 < 0) or (l0 > map.lmax):
        return -np.inf
    else:
        ll = 0

    # Compute the GP prior
    for l in range(0, lmax + 1):
        kernel = power(l, amp=amp, alpha=alpha, beta=beta, l0=l0) ** 2 * \
                 kernels.Matern32Kernel(tau ** 2)
        gp = george.GP(kernel)
        gp.compute(time)
        for m in range(-l, l + 1):
            ll += gp.log_likelihood(y[l ** 2 + l + m, :])

    # Compute the likelihood
    ll += -0.5 * np.sum((model - flux) ** 2) / err ** 2

    return ll


def neglnlike(*args, **kwargs):
    return -lnlike(*args, **kwargs)


example()
quit()


# These are fixed
lmax = 5
npts = 300
err = 0.01
time = np.linspace(0.0, 1.0, npts)

# Parameters we'll try to recover
per = 0.4345
amp = 0.01
tau = 0.125
alpha = 2.0
beta = -2.0
l0 = 1

# Generate the synthetic light curve
y = variable_map(lmax=lmax, amp=amp, tau=tau, npts=npts,
                 alpha=alpha, beta=beta, l0=l0)
map = starry.Map(lmax, npts)
map[:, :] = y
flux0 = phase_curve(map, time, per)
flux = flux0 + err * np.random.randn(npts)

x = np.array([amp, tau, alpha, beta, l0, per] + list(y.flatten()))
print(x.shape)
print(lnlike(x, time, flux, err, map))
