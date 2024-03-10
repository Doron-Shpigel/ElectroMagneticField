
import sympy as sp
from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt
from sympy import sin, cos, acos, atan2, sqrt, exp, pi, symbols, UnevaluatedExpr, latex, Piecewise, Function, solve, Derivative
from sympy.vector import Vector, cross, CoordSys3D, ParametricRegion, ImplicitRegion, vector_integrate, Del, curl, express, divergence, gradient, ParametricIntegral
from sympy.geometry import Point, Circle, Triangle
from sympy import init_printing
from sympy.printing import latex
from sympy import Integral, integrate
from numpy import meshgrid, ravel


#Initial coordinate systems setup
C = CoordSys3D('C') #(x, y, z)
P = C.create_new('P', transformation='cylindrical', variable_names = ["r", "phi", "z"]) #r, phi, z: (r*cos(phi), r*sin(phi), z)
S = C.create_new('S', transformation='spherical') #r, theta, phi: (r*sin(theta)*cos(phi), r*sin(theta)*sin(phi),r*cos(theta))

from src.VectorPathIntegral import vector_path_integral, vector_circle_integral
from src.Vectors import NewVector, NewVectorFunction, PrintVector, VectorMesh
from src.SpecialOperators import change_derivatives_to_constant_by_denominator, XYZ_to_Function
