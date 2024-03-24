import os


from src.Vectors import coords, VectorMesh, Cartesian_to_Spherical, Cartesian_to_Cylindrical
from src.SpecialOperators import XYZ_to_Function

from numpy import linspace

C, P, S = coords()


def simpleField(V, A, B):
    return (A*B/(3*V.coeff(P.i)))*P.j


x = linspace(-5, 5, 10)
y = linspace(-5, 5, 10)
z = linspace(0, 2, 4)

H = VectorMesh(x, y, z)
H.field(P, simpleField, A = 1, B = 1)


data = H.plotlyConeData()
import plotly.graph_objs as go
layout = go.Layout(title=r'H field ',
                    scene=dict(xaxis_title=r'x',
                                yaxis_title=r'y',
                                zaxis_title=r'z',
                                aspectratio=dict(x=1, y=1, z=1),
                                camera_eye=dict(x=1.2, y=1.2, z=1.2)))

fig = go.Figure(data = data, layout=layout)
filename = "test_vectormesh.html"
current_directory = os.getcwd()
test_directory = os.path.join(current_directory, 'tests')
export_subdirectory = os.path.join(test_directory, 'tests_export')
if not os.path.exists(export_subdirectory):
    os.makedirs(export_subdirectory)

final_directory = os.path.join(export_subdirectory, filename)
fig.write_html(final_directory)



