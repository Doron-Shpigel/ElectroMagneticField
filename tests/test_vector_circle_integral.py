#TODO: test run of vector_circle_integral function, and compere it sympy vector_integrate function
import sys
import os

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from Vectors import coords, VectorMesh, Cartesian_to_Spherical, Cartesian_to_Cylindrical
from SpecialOperators import XYZ_to_Function, vector_circle_integral, vector_path_integral

from numpy import linspace
import sympy
import sympy.vector as vec

C, P, S = coords()

x, y, t, r = sympy.symbols('x y t r')
test_vector01 =  C.x*C.i + C.y*C.j
test_curve01 = vec.ParametricRegion((sympy.cos(t), sympy.sin(t)), (t, sympy.pi,0))
result01 = vec.vector_integrate(test_vector01, test_curve01)
print(result01) # = 0 reference test
test_vector02 = P.r*P.i # vector02 is vector01 but in polar coordinates 
test_curve02 = vec.ParametricRegion((1, t, 0), (t, sympy.pi, 0))
result02 = vec.vector_integrate(test_vector02, test_curve02)
print(result02) # = 0 reference test
test_vector02 = P.r*P.j
test_curve02 = vec.ParametricRegion((r, t, 0), (r, 1, 1), (t, sympy.pi, 0))
result02 = vec.vector_integrate(test_vector02, test_curve02)
print(result02) # 0  = Wrong, did not used the correct length element: dl = dr*P.i + r*d_phi*P.j + dz*P.k
result03 = vector_path_integral(test_vector02, (P.r, 1,1),
                                    (P.phi, sympy.pi,0), (P.z, 0, 0))
print(result03) # result03 = 0 - pi*P.r**2 = -pi*P.r**2
radius = sympy.symbols('radius', positive=True, real=True)
result04 = vector_path_integral(test_vector02, (radius, 1,1),
                                    (P.phi, sympy.pi,0), (P.z, 0, 0))
print(result04) # result04 = 0 - pi*radius**2 = -pi*radius**2
result05 = vector_path_integral(test_vector02, radius,
                                    (P.phi, sympy.pi,0), (P.z, 0, 0))
print(result05) # result05 = result04 --> can handle radius as a symbol

test_vector03 = P.r*P.i + P.phi*P.j
result06 = vector_circle_integral(test_vector03, radius)
print(vec.is_conservative(test_vector03)) # = False
print(result06) # result06 = 0 - pi*radius**2 = -pi*radius**2
test_vector04 = C.y**2*C.z**3*C.i + 2*C.x*C.y*C.z**3*C.j + 3*C.x*C.y**2*C.z**2*C.k
print(vec.is_conservative(test_vector04)) # = True
result07 = vector_circle_integral(test_vector04, radius)
print(result07) # result07 = 0 
