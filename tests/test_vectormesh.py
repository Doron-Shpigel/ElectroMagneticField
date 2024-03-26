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


data1 = H.plotlyConeData()
import plotly.graph_objs as go
layout1 = go.Layout(title=r'H field ',
                    scene=dict(xaxis_title=r'x',
                                yaxis_title=r'y',
                                zaxis_title=r'z',
                                aspectratio=dict(x=1, y=1, z=1),
                                camera_eye=dict(x=1.2, y=1.2, z=1.2)))

fig1 = go.Figure(data = data1, layout=layout1)
filename = "test_vectormesh_cone.html"
current_directory = os.getcwd()
test_directory = os.path.join(current_directory, 'tests')
export_subdirectory = os.path.join(test_directory, 'tests_export')
if not os.path.exists(export_subdirectory):
    os.makedirs(export_subdirectory)

final_directory = os.path.join(export_subdirectory, filename)
fig1.write_html(final_directory)

data2 = H.plotlyArrowData(plane="xy")
layout2 = go.Layout(
    title="2D Vector Field original",
    xaxis=dict(title="X"),
    yaxis=dict(title="Y"),
    legend=dict(orientation="h")
)
fig2 = go.Figure(data = data2, layout=layout2)
filename = "test_vectormesh_arrow_xy.html"
final_directory = os.path.join(export_subdirectory, filename)
fig2.write_html(final_directory)

data3 = H.plotlyArrowData(plane="xz")
layout3 = go.Layout(
    title="2D Vector Field original",
    xaxis=dict(title="X"),
    yaxis=dict(title="Z"),
    legend=dict(orientation="h")
)
fig3 = go.Figure(data = data3, layout=layout3)
filename = "test_vectormesh_arrow_xz.html"
final_directory = os.path.join(export_subdirectory, filename)
fig3.write_html(final_directory)

data4 = H.plotlyArrowData(plane="yz")
layout4 = go.Layout(
    title="2D Vector Field original",
    xaxis=dict(title="Y"),
    yaxis=dict(title="Z"),
    legend=dict(orientation="h")
)
fig4 = go.Figure(data = data4, layout=layout4)
filename = "test_vectormesh_arrow_yz.html"
final_directory = os.path.join(export_subdirectory, filename)
fig4.write_html(final_directory)

