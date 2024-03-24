import numpy as np
import matplotlib.pyplot as plt


def plot_polar_heatmap(func, Title='', Label = '', radius=1, theta=2*np.pi, precision = 200):
    """
    Plot a heatmap of a function in polar coordinates
    """
    R = np.linspace(0, radius*1.1, precision)
    Theta = np.linspace(0, theta, precision)
    R_grid, Theta_grid = np.meshgrid(R, Theta)
    heat = np.zeros_like(R_grid)
    for i in range(len(R)):
        for j in range(len(Theta)):
            heat[j,i] = func(R[i], Theta[j])
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_title(Title)
    c = ax.pcolormesh(Theta_grid, R_grid, heat, cmap='coolwarm', label=Label)
    cbar = fig.colorbar(c, ax=ax, label=Label)
    return fig, ax