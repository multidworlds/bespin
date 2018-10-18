import starry
import matplotlib.pyplot as pl
import matplotlib.animation as animation
import numpy as np
import george
from george import kernels
from tqdm import tqdm
np.random.seed(42)
cmap = pl.get_cmap("inferno")


def power(l, alpha=2.0, beta=-2.0, l0=2):
    """A broken power law power spectrum."""
    if hasattr(l, "__len__"):
        p = ((l + 1.0) / (l0 + 1.0)) ** alpha
        p[l >= l0] = ((l[l >= l0] + 1.0) / (l0 + 1.0)) ** beta
        return p
    else:
        if l < l0:
            return ((l + 1.0) / (l0 + 1)) ** alpha
        else:
            return ((l + 1.0) / (l0 + 1)) ** beta


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


def variable_map(lmax=10, amp=1.0, tau=0.025, npts=300,
                 alpha=2.0, beta=-2.0, l0=2):
    """Generate variability using a GP."""
    time = np.linspace(0.0, 1.0, npts)
    y = np.zeros(((lmax + 1) ** 2, npts))
    n = 0
    for l in range(0, lmax + 1):
        if hasattr(tau, "__len__"):
            kernel = amp * power(l, alpha=alpha, beta=beta, l0=l0) ** 2 * \
                     kernels.Matern32Kernel([tau[0] ** 2,
                        (tau[1] * (2 * l + 1)) ** 2], ndim=2)
            x = np.array([[ti, m] for ti in time for m in range(-l, l + 1)])
            gp = george.GP(kernel)
            sample = gp.sample(x).reshape(-1, 2 * l + 1).transpose()
        else:
            kernel = amp * power(l, alpha=alpha, beta=beta, l0=l0) ** 2 * \
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


# Create a variable map
lmax = 10
amp = 1.0
tau = 0.03
npts = 300
nrot = 3.0
alpha = 2.0
beta = -2.0
l0 = 2
y = variable_map(lmax=lmax, amp=amp, tau=tau, npts=npts,
                 alpha=alpha, beta=beta, l0=l0)
fig, ax = plot_power(y)
fig.savefig("powerspec.png", bbox_inches="tight")
fig, ax = plot_coeffs(y)
fig.savefig("coeffs.png", bbox_inches="tight")
animate(y, nrot=nrot, gif="clouds.gif")
