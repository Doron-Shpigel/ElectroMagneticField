import os
from sympy import symbols, Piecewise
from src.Plots import plot_polar_heatmap
import numpy as np
# Define constants




# Define the current density function in cylindrical coordinates
def current_density_cylindrical(r, phi, z=0):
    return Piecewise((A* (r / R), r < R), (0, True))
R = 4
A = 1
fig, ax = plot_polar_heatmap(current_density_cylindrical, Title = 'Current Density in Cylindrical Coordinates', Label = r'$J(r, \phi)$', radius = 5, theta = 2*np.pi, precision = 200)
filename = "test_Plots.png"
current_directory = os.getcwd()
test_directory = os.path.join(current_directory, 'tests')
export_subdirectory = os.path.join(test_directory, 'tests_export')
if not os.path.exists(export_subdirectory):
    os.makedirs(export_subdirectory)

final_directory = os.path.join(export_subdirectory, filename)
fig.savefig(final_directory)
