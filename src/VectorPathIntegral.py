import sympy as sp
from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt
from sympy import sin, cos, atan2,, sqrt, exp, pi, symbols, UnevaluatedExpr, latex, Piecewise, Function, solve, Derivative
from sympy.vector import cross, CoordSys3D, ParametricRegion, ImplicitRegion, vector_integrate, Del, curl, express, divergence, gradient, ParametricIntegral
from sympy.geometry import Point, Circle, Triangle
from sympy import init_printing
from sympy.printing import latex
from sympy import Integral, integrate


def vector_path_integral(V, cor1, cor2, cor3):
  system = V._sys
  if system == P: #cylindrical
    r_int= Integral((V.coeff(P.i)), cor1).doit()
    if isinstance(cor1, tuple):
      phi_int = Integral((cor1[0]*V.coeff(P.j)), cor2).doit()
    else:
      phi_int = Integral((cor1*V.coeff(P.j)), cor2).doit()
    z_int = Integral((V.coeff(P.k)), cor3).doit()
    return r_int + phi_int + z_int
  elif system == S:
    r_int= Integral((V.coeff(S.i)), cor1).doit()
    cor1sym = cor1
    cor2sym = cor2
    if isinstance(cor1, tuple):
      cor1sym = cor1[0]
    if isinstance(cor2, tuple):
      cor2sym = cor2[0]
    theta_int = Integral((cor1sym*V.coeff(S.j)), cor2).doit()
    phi_int = Integral((cor1sym*sin(cor2sym)*V.coeff(S.k)), cor3).doit()
    return r_int + theta_int + phi_int
  else: #system is C
    x_int = Integral((V.coeff(C.i)), cor1).doit()
    y_int = Integral((V.coeff(C.j)), cor2).doit()
    z_int = Integral((V.coeff(C.k)), cor3).doit()
    return x_int + y_int + z_int


def vector_circle_integral(V, radius):
    system = V._sys
    if system == P:
      return vector_path_integral(V, radius, (P.phi, 0, 2*pi), (P.z, 0, 0))
    elif system == S:
      return vector_path_integral(V, radius, (P.phi, 0, 2*pi), (S.theta, 0, 0))
    else: #system is C
      x = radius * cos(t)
      y = radius * sin(t)
      t= symbols('t')
      return vector_path_integral(V.subs({x: radius * cos(t), y: radius * sin(t)}), (t, 0, 2 * pi), (t, 0, 2 * pi), (t, 0, 2 * pi))
