import sympy as sp
from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt
from sympy import sin, cos, atan2, sqrt, exp, pi, symbols, UnevaluatedExpr, latex, Piecewise, Function, solve, Derivative
from sympy.vector import cross, CoordSys3D, ParametricRegion, ImplicitRegion, vector_integrate, Del, curl, express, divergence, gradient, ParametricIntegral
from sympy.geometry import Point, Circle, Triangle
from sympy import init_printing
from sympy.printing import latex
from sympy import Integral, integrate

#Initial coordinate systems setup
C = CoordSys3D('C') #(x, y, z)
P = C.create_new('P', transformation='cylindrical', variable_names = ["r", "phi", "z"]) #r, phi, z: (r*cos(phi), r*sin(phi), z)
S = C.create_new('S', transformation='spherical') #r, theta, phi: (r*sin(theta)*cos(phi), r*sin(theta)*sin(phi),r*cos(theta))


def PrintVector(V):
    from sympy.vector import Vector
    if isinstance(V, Vector):
        system = V._sys
        if system == C:
            str_vector_i = rf"\left[" + latex(V.coeff(C.i)) + rf"\right] \hat{{x}}"
            str_vector_j = rf"\left[" + latex(V.coeff(C.j)) + rf"\right] \hat{{y}}"
            str_vector_k = rf"\left[" + latex(V.coeff(C.k)) + rf"\right] \hat{{z}}"
            str_vector =  str_vector_i + "+" + str_vector_j + "+" + str_vector_k
            str_vector = str_vector.replace(r"\mathbf{{x}_{C}}", "x")
            str_vector = str_vector.replace(r"\mathbf{{y}_{C}}", "y")
            str_vector = str_vector.replace(r"\mathbf{{z}_{C}}", "z")
            return str_vector
        elif system == P:
            str_vector_i = rf"\left[" + latex(V.coeff(P.i)) + rf"\right] \hat{{r}}"
            str_vector_j = rf"\left[" + latex(V.coeff(P.j)) + rf"\right] \hat{{\phi}}"
            str_vector_k = rf"\left[" + latex(V.coeff(P.k)) + rf"\right] \hat{{z}}"
            str_vector =  str_vector_i + "+" + str_vector_j + "+" + str_vector_k
            str_vector = str_vector.replace(r"\mathbf{{r}_{P}}", "r")
            str_vector = str_vector.replace(r"\mathbf{{phi}_{P}}", "\phi")
            str_vector = str_vector.replace(r"\mathbf{{z}_{P}}", "z")
            return str_vector
        elif system == S:
            str_vector_i = rf"\left[" + latex(V.coeff(S.i)) + rf"\right] \hat{{r}}"
            str_vector_j = rf"\left[" + latex(V.coeff(S.j)) + rf"\right] \hat{{\theta}}"
            str_vector_k = rf"\left[" + latex(V.coeff(S.k)) + rf"\right] \hat{{\phi}}"
            str_vector =  str_vector_i + "+" + str_vector_j + "+" + str_vector_k
            str_vector = str_vector.replace(r"\mathbf{{r}_{S}}", "r")
            str_vector = str_vector.replace(r"\mathbf{{theta}_{S}}", "\theta")
            str_vector = str_vector.replace(r"\mathbf{{phi}_{S}}", "\phi")
            return str_vector
    else:
        x, y, z = symbols('x y z')
        V = V.subs(C.x, UnevaluatedExpr(x))
        V = V.subs(C.y, UnevaluatedExpr(y))
        V = V.subs(C.z, UnevaluatedExpr(z))
        r, theta, phi = symbols('r theta phi')
        V = V.subs(S.r, UnevaluatedExpr(r))
        V = V.subs(S.theta, UnevaluatedExpr(theta))
        V = V.subs(S.phi, UnevaluatedExpr(phi))
        r, phi, z = symbols('r phi z')
        V = V.subs(P.r, UnevaluatedExpr(r))
        V = V.subs(P.phi, UnevaluatedExpr(phi))
        V = V.subs(P.z, UnevaluatedExpr(z))
        return latex(V)

def NewVector(name, system):
    if system == C:
        a = Function(f"{name}x")(C.x, C.y, C.z)
        b = Function(f"{name}y")(C.x, C.y, C.z)
        c = Function(f"{name}z")(C.x, C.y, C.z)
        return a*C.i + b*C.j + c*C.k
    elif system == P:
        a = Function(f"{name}r")(P.r, P.phi, P.z)
        b = Function(f"{name}phi")(P.r, P.phi, P.z)
        c = Function(f"{name}z")(P.r, P.phi, P.z)
        return a*P.i + b*P.j + c*P.k
    elif system == S:
        a = Function(f"{name}r")(S.r, S.theta, S.phi)
        b = Function(f"{name}theta")(S.r, S.theta, S.phi)
        c = Function(f"{name}phi")(S.r, S.theta, S.phi)
        return a*S.i + b*S.j + c*S.k

class NewVectorFunction:

    def __init__(self, name, system):
        self.name = name
        self.system = system
        if system == C:
            self.i = Function(f"{self.name}x")(C.x, C.y, C.z)
            self.j = Function(f"{self.name}y")(C.x, C.y, C.z)
            self.k = Function(f"{self.name}z")(C.x, C.y, C.z)
            #self.vector = self.i*C.i + self.j*C.j + self.k*C.k
        elif system == P:
            self.i = Function(f"{self.name}r")(P.r, P.phi, P.z)
            self.j = Function(f"{self.name}phi")(P.r, P.phi, P.z)
            self.k = Function(f"{self.name}z")(P.r, P.phi, P.z)
            #self.vector = self.i*P.i + self.j*P.j + self.k*P.k
        elif system == S:
            self.i = Function(f"{self.name}r")(S.r, S.theta, S.phi)
            self.j = Function(f"{self.name}theta")(S.r, S.theta, S.phi)
            self.k = Function(f"{self.name}phi")(S.r, S.theta, S.phi)
            #self.vector = self.i*S.i + self.j*S.j + self.k*S.k
        self.update_vector()

    def update_vector(self):
        self.vector = self.i*S.i + self.j*S.j + self.k*S.k

    def __call__(self):
        self.update_vector()
        return self.vector
    def __str__(self):
        self.update_vector()
        return printVector(self.vector)
    def __repr__(self):
        self.update_vector()
        return self.vector


